# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoder

sq_len, epochs = 15, 30
b_size, v_size = 128, 20000
e_dim, h_dim, z_dim = 128, 128, 16
an_max, an_bias = 1., 6000.
dropout, clip, beam = .8, 5., 5

class ModifiedBasicDecoder(BasicDecoder):
	def __init__(self, cell, helper, initial_state, concat_z, output_layer=None):
		super(ModifiedBasicDecoder,self).__init__(cell,helper,initial_state,output_layer)
		self.z = concat_z
	def initialize(self, name=None):
		finished, first_inputs, initial_state = super(ModifiedBasicDecoder,self).initialize(name)
		first_inputs = array_ops.concat([first_inputs,self.z],-1)
		return finished, first_inputs, initial_state
	def step(self, time, inputs, state, name=None):
		outputs, next_state, next_inputs, finished = super(ModifiedBasicDecoder,self).step(time,inputs,state,name)
		next_inputs = array_ops.concat([next_inputs,self.z],-1)
		return outputs, next_state, next_inputs, finished

class ModifiedBeamSearchDecoder(BeamSearchDecoder):
	def __init__(self, cell, embedding, start_tokens, end_token, initial_state, beam_width, concat_z, output_layer=None, length_penalty_weight=0.0):
		super(ModifiedBeamSearchDecoder,self).__init__(cell,embedding,start_tokens,end_token,initial_state,beam_width,output_layer,length_penalty_weight)
		self.z = concat_z
	def initialize(self, name=None):
		finished, start_inputs, initial_state = super(ModifiedBeamSearchDecoder,self).initialize(name)
		start_inputs = array_ops.concat([start_inputs,self.z],-1)
		return finished, start_inputs, initial_state
	def step(self, time, inputs, state, name=None):
		beam_search_output, beam_search_state, next_inputs, finished = super(ModifiedBeamSearchDecoder,self).step(time,inputs,state,name)
		next_inputs = array_ops.concat([next_inputs,self.z],-1)
		return beam_search_output, beam_search_state, next_inputs, finished

def forward(inputs, labels, mode):
	def rnn_cell():
		return tf.nn.rnn_cell.GRUCell(h_dim,kernel_initializer=tf.orthogonal_initializer())
	def reparam_trick(z_m, z_v):
		return z_m+tf.exp(0.5*z_v)*tf.truncated_normal(tf.shape(z_v))
	enc_sq_len = tf.count_nonzero(inputs,1,dtype=tf.int32)
	batch_size = tf.shape(inputs)[0]
	with tf.variable_scope('encoder'):
		embed = tf.get_variable('embed',[v_size,e_dim])
		x = tf.nn.embedding_lookup(embed,inputs)
		_,enc_state = tf.nn.dynamic_rnn(rnn_cell(),x,enc_sq_len,dtype=tf.float32)
		z_m = tf.layers.dense(enc_state,z_dim)
		z_v = tf.layers.dense(enc_state,z_dim)  
	z = reparam_trick(z_m,z_v)
	with tf.variable_scope('decoder'):
		init_state = tf.layers.dense(z,h_dim,tf.nn.elu)
		out_proj = tf.layers.Dense(v_size,_scope='decoder/output_proj')
		dec_cell = rnn_cell()
		if mode == tf.estimator.ModeKeys.TRAIN:
			dec_sq_len = tf.count_nonzero(labels['dec_out'],1,dtype=tf.int32)
			helper = tf.contrib.seq2seq.TrainingHelper(
				inputs=tf.nn.embedding_lookup(embed,labels['dec_in']),
				sequence_length=dec_sq_len)
			dec = ModifiedBasicDecoder(
				cell=dec_cell,
				helper=helper,
				initial_state=init_state,
				concat_z=z)
			dec_out,_,_ = tf.contrib.seq2seq.dynamic_decode(
				decoder=dec,
				maximum_iterations=tf.reduce_max(dec_sq_len))
			rnn_out = dec_out.rnn_output
			return rnn_out,out_proj(rnn_out),z_m,z_v
		else:
			tiled_z = tf.tile(tf.expand_dims(z,1),[1,beam,1])
			dec = ModifiedBeamSearchDecoder(
				cell=dec_cell,
				embedding=embed,
				start_tokens=tf.tile(tf.constant([wd2id['<start>']],tf.int32),[batch_size]),
				end_token=wd2id['<end>'],
				initial_state=tf.contrib.seq2seq.tile_batch(init_state,beam),
				beam_width=beam,
				output_layer=out_proj,
				concat_z=tiled_z)
			dec_out,_,_ = tf.contrib.seq2seq.dynamic_decode(
				decoder=dec)
			return dec_out.predicted_ids[:,:,0]

def model_fn(features, labels, mode):
	def clip_grads(loss):
		variables = tf.trainable_variables()
		return zip(tf.clip_by_global_norm(tf.gradients(loss,variables),clip)[0],variables)
	logits_or_ids = forward(features,labels,mode)        
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode,predictions=logits_or_ids)
	if mode == tf.estimator.ModeKeys.TRAIN:
		rnn_out,logits,z_m,z_v = logits_or_ids
		global_step = tf.train.get_global_step()
		mask = tf.reshape(tf.to_float(tf.sign(labels['dec_out'])),[-1])
		with tf.variable_scope('decoder/decoder/output_proj',reuse=True):
			_W = tf.transpose(tf.get_variable('kernel'))
			_b = tf.get_variable('bias')
		nll_loss = tf.reduce_sum(mask*tf.nn.sampled_softmax_loss(
			weights=_W,
			biases=_b,
			labels=tf.reshape(labels['dec_out'],[-1,1]),
			inputs=tf.reshape(rnn_out,[-1,h_dim]),
			num_sampled=1000,
			num_classes=v_size,
 		))/tf.to_float(tf.shape(features)[0])
		kl_loss = 0.5*tf.reduce_sum(tf.exp(z_v)+tf.square(z_m)-1-z_v)/tf.to_float(tf.shape(z_m)[0])
		alpha = an_max*tf.sigmoid((10/an_bias)*(tf.to_float(global_step)-tf.constant(an_bias/2)))
		loss_op = nll_loss+alpha*kl_loss
		train_op = tf.train.AdamOptimizer().apply_gradients(clip_grads(loss_op),global_step=global_step)
		lth = tf.train.LoggingTensorHook({'nll_loss':nll_loss,'kl_loss':kl_loss,'alpha':alpha},every_n_iter=100)
		return tf.estimator.EstimatorSpec(mode=mode,loss=loss_op,train_op=train_op,training_hooks=[lth])
		
if __name__ == '__main__':
	def load_vocab(start=4):
		wd2id = {k:v+start for k,v in tf.keras.datasets.imdb.get_word_index().iteritems()}
		wd2id.update({'<pad>':0,'<start>':1,'<unk>':2,'<end>':3})
		id2wd = {i:w for w,i in wd2id.iteritems()}
		return wd2id,id2wd
	def load_data(start=4):
		(X_tr,_),(X_te,_) = tf.contrib.keras.datasets.imdb.load_data(num_words=v_size,index_from=start)
		return X_tr,X_te
	def proc_input(strs):
		x = [[wd2id.get(w,2) for w in s.split()] for s in strs]
		return tf.keras.preprocessing.sequence.pad_sequences(x,sq_len,truncating='post',padding='post')
	def word_dropout(x):
		return np.vectorize(lambda x,k: wd2id['<unk>'] if k and x>=4 else x)(\
			x,np.random.binomial(1,dropout,x.shape))
	def demo(test_strs, pred_ids):
		for test,pred in zip(test_strs,pred_ids):
			print 'orig:',test,
			print 'reco:',' '.join([id2wd.get(i,'<unk>') for i in pred])
	wd2id,id2wd = load_vocab()
	X = np.concatenate(load_data())
	X = np.concatenate((tf.keras.preprocessing.sequence.pad_sequences(X,sq_len,truncating='post',padding='post'),
						tf.keras.preprocessing.sequence.pad_sequences(X,sq_len,truncating='pre',padding='post')))
	enc_in, dec_in = X[:,1:], X[:]
	dec_out = np.concatenate([X[:,1:],np.full([X.shape[0],1],wd2id['<end>'])],1)
	test_strs = ['i love this film and i think it is one of the best films',
				 'this movie is a waste of time and there is no point to watch it']
	estimator = tf.estimator.Estimator(model_fn)
	for i in xrange(epochs):
		print 'Epoch {0}/{1}'.format(i+1,epochs)        
		estimator.train(tf.estimator.inputs.numpy_input_fn(
			x=enc_in,
			y={'dec_in':word_dropout(dec_in),'dec_out':dec_out},
			batch_size=b_size,
			shuffle=True))
		pred_ids = list(estimator.predict(tf.estimator.inputs.numpy_input_fn(
			x=proc_input(test_strs),
			shuffle=False)))
		demo(test_strs,pred_ids)
