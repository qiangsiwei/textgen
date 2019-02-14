# -*- coding: utf-8 -*-

import numpy as np
from gan import Gan
from utils import *
from text_proc import *
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Generator(object):
	def __init__(self, v_size, b_size, e_dim, h_dim, sq_len, start, lr=.01):
		self.v_size = v_size
		self.b_size = b_size
		self.e_dim = e_dim
		self.h_dim = h_dim
		self.sq_len = sq_len
		self.start = tf.constant([start]*b_size,dtype=tf.int32)
		self.lr = lr
		self.clip = 5.0
		self.g_params = []
		def init_matrix(shape):
			return tf.random_normal(shape,stddev=0.1)
		def recurrent_unit(params):
			self.Wi = tf.Variable(init_matrix([e_dim,h_dim]))
			self.Ui = tf.Variable(init_matrix([h_dim,h_dim]))
			self.bi = tf.Variable(init_matrix([h_dim]))
			self.Wf = tf.Variable(init_matrix([e_dim,h_dim]))
			self.Uf = tf.Variable(init_matrix([h_dim,h_dim]))
			self.bf = tf.Variable(init_matrix([h_dim]))
			self.Wg = tf.Variable(init_matrix([e_dim,h_dim]))
			self.Ug = tf.Variable(init_matrix([h_dim,h_dim]))
			self.bg = tf.Variable(init_matrix([h_dim]))
			self.Wc = tf.Variable(init_matrix([e_dim,h_dim]))
			self.Uc = tf.Variable(init_matrix([h_dim,h_dim]))
			self.bc = tf.Variable(init_matrix([h_dim]))
			params.extend([
				self.Wi,self.Ui,self.bi,
				self.Wf,self.Uf,self.bf,
				self.Wg,self.Ug,self.bg,
				self.Wc,self.Uc,self.bc])
			def unit(x, h_mem):
				h_prev, c_prev = tf.unstack(h_mem)
				i = tf.sigmoid(tf.matmul(x,self.Wi)+tf.matmul(h_prev,self.Ui)+self.bi)
				f = tf.sigmoid(tf.matmul(x,self.Wf)+tf.matmul(h_prev,self.Uf)+self.bf)
				o = tf.sigmoid(tf.matmul(x,self.Wg)+tf.matmul(h_prev,self.Ug)+self.bg)
				c = f*c_prev+i*tf.nn.tanh(tf.matmul(x,self.Wc)+tf.matmul(h_prev,self.Uc)+self.bc)
				h = o*tf.nn.tanh(c)
				return tf.stack([h,c])
			return unit
		def output_unit(params):
			self.Wo = tf.Variable(init_matrix([h_dim,v_size]))
			self.bo = tf.Variable(init_matrix([v_size]))
			params.extend([self.Wo,self.bo])
			def unit(h_mem):
				h_prev, c_prev = tf.unstack(h_mem)
				logits = tf.matmul(h_prev,self.Wo)+self.bo
				return logits
			return unit
		with tf.variable_scope('generator'):
			self.g_embed = tf.Variable(init_matrix([v_size,e_dim]))
			self.g_params.append(self.g_embed)
			self.g_runit = recurrent_unit(self.g_params)
			self.g_ounit = output_unit(self.g_params)
		# generate
		self.h0 = tf.zeros([b_size,h_dim])
		self.h0 = tf.stack([self.h0,self.h0]) # h&c
		gen_o = tensor_array_ops.TensorArray(dtype=tf.float32,size=sq_len,dynamic_size=False,infer_shape=True)
		gen_x = tensor_array_ops.TensorArray(dtype=tf.int32,size=sq_len,dynamic_size=False,infer_shape=True)
		def recurrent(i, x_t, h_p, gen_o, gen_x):
			h_t = self.g_runit(x_t,h_p)
			o_t = self.g_ounit(h_t)
			prob = tf.log(tf.nn.softmax(o_t))
			next = tf.cast(tf.reshape(tf.multinomial(prob,1),[b_size]),tf.int32)
			x_n = tf.nn.embedding_lookup(self.g_embed,next)
			gen_o = gen_o.write(i,tf.reduce_sum(tf.multiply(tf.one_hot(next,v_size,1.0,0.0),tf.nn.softmax(o_t)),1))
			gen_x = gen_x.write(i,next)
			return i+1, x_n, h_t, gen_o, gen_x
		_,_,_,self.gen_o,self.gen_x = control_flow_ops.while_loop(
			cond=lambda i,_1,_2,_3,_4:i<sq_len, body=recurrent,
			loop_vars=(tf.constant(0,dtype=tf.int32),
				tf.nn.embedding_lookup(self.g_embed,self.start),self.h0,gen_o,gen_x))
		self.gen_x = self.gen_x.stack()
		self.gen_x = tf.transpose(self.gen_x,perm=[1,0])
		# pretrain
		self.x = tf.placeholder(tf.int32,shape=[b_size,sq_len])
		self.r = tf.placeholder(tf.float32,shape=[b_size,sq_len])
		self.emb_x = tf.transpose(tf.nn.embedding_lookup(self.g_embed,self.x),perm=[1,0,2])
		g_pred = tensor_array_ops.TensorArray(dtype=tf.float32,size=sq_len,dynamic_size=False,infer_shape=True)
		tf_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32,size=sq_len) # teacher forcing
		tf_emb_x = tf_emb_x.unstack(self.emb_x)
		def pretrain_recurrent(i, x_t, h_p, g_preds):
			h_t = self.g_runit(x_t,h_p)
			o_t = self.g_ounit(h_t)
			g_preds = g_preds.write(i,tf.nn.softmax(o_t))
			x_n = tf_emb_x.read(i)
			return i+1, x_n, h_t, g_preds	
		_,_,_,self.g_preds = control_flow_ops.while_loop(
			cond=lambda i,_1,_2,_3:i<sq_len, body=pretrain_recurrent,
			loop_vars=(tf.constant(0,dtype=tf.int32),
				tf.nn.embedding_lookup(self.g_embed,self.start),self.h0,g_pred))
		self.g_preds = tf.transpose(self.g_preds.stack(),perm=[1,0,2])
		self.pretrain_loss = -tf.reduce_sum(\
			tf.one_hot(tf.to_int32(tf.reshape(self.x,[-1])),v_size,1.0,0.0)*\
			tf.log(tf.clip_by_value(tf.reshape(self.g_preds,[-1,v_size]),1e-20,1.0))
			)/(sq_len*b_size)
		def g_optimizer(self, *args, **kwargs):
			return tf.train.AdamOptimizer(*args, **kwargs)
		pretrain_opt = g_optimizer(lr)
		self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss,self.g_params),self.clip)
		self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad,self.g_params))
		# reinforcement learning
		self.g_loss = -tf.reduce_sum(
			tf.reduce_sum(
				tf.one_hot(tf.to_int32(tf.reshape(self.x,[-1])),v_size,1.0,0.0)*\
				tf.log(tf.clip_by_value(tf.reshape(self.g_preds,[-1,v_size]),1e-20,1.0)),
			1)*tf.reshape(self.r,[-1]))
		g_opt = g_optimizer(lr)
		self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss,self.g_params),self.clip)
		self.g_updates = g_opt.apply_gradients(zip(self.g_grad,self.g_params))
	def pretrain(self, sess, x):
		return sess.run([self.pretrain_updates,self.pretrain_loss],feed_dict={self.x:x})
	def generate(self, sess):
		return sess.run(self.gen_x)

class Discriminator(object):
	def __init__(self, v_size, sq_len, e_dim, f_size, f_num, lr=.01, l2_lam=0., dropout=1.):
		self.in_x = tf.placeholder(tf.int32,[None,sq_len])
		self.in_y = tf.placeholder(tf.float32,[None,2])
		def linear(x, out_size, scope='linear'):
			shape = x.get_shape().as_list()
			assert len(shape)==2 and shape[1]
			with tf.variable_scope(scope):
				M = tf.get_variable('M',[out_size,shape[1]],dtype=x.dtype)
				b = tf.get_variable('b',[out_size],dtype=x.dtype)
			return tf.matmul(x,tf.transpose(M))+b
		def highway(x, size, layers=1, b=-2.0, f=tf.nn.relu, scope='highway'):
			with tf.variable_scope(scope):
				for i in xrange(layers):
					g = f(linear(x,size,scope='highway_linear_{}'.format(i)))
					t = tf.sigmoid(linear(x,size,scope='highway_gate_{}'.format(i))+b)
					out = t*g+(1.-t)*x; x = out
			return out
		with tf.variable_scope('discriminator'):
			self.W = tf.Variable(tf.random_uniform([v_size,e_dim],-1.0,1.0))
			self.embed_in_x = tf.expand_dims(tf.nn.embedding_lookup(self.W,self.in_x),-1)
			pool_outs = []
			for f_s, f_n in zip(f_size, f_num):
				with tf.name_scope('conv-maxpool-{}'.format(f_s)):
					W = tf.Variable(tf.truncated_normal([f_s,e_dim,1,f_n],stddev=0.1))
					b = tf.Variable(tf.constant(0.1,shape=[f_n]))
					conv = tf.nn.conv2d(self.embed_in_x,W,strides=[1,1,1,1],padding='VALID')
					h = tf.nn.relu(tf.nn.bias_add(conv,b))
					pool = tf.nn.max_pool(h,ksize=[1,sq_len-f_s+1,1,1],strides=[1,1,1,1],padding='VALID')
					pool_outs.append(pool)
			self.h_pool = tf.reshape(tf.concat(pool_outs,3),[-1,sum(f_num)])
			with tf.name_scope('highway'):
				self.h_highway = highway(self.h_pool,self.h_pool.get_shape()[1],1,0)
			with tf.name_scope('dropout'):
				self.h_drop = tf.nn.dropout(self.h_highway,dropout)
			with tf.name_scope('output'):
				W = tf.Variable(tf.truncated_normal([sum(f_num),2],stddev=0.1))
				b = tf.Variable(tf.constant(0.1,shape=[2]))
				l2_loss = tf.nn.l2_loss(W)+tf.nn.l2_loss(b)
				self.scores = tf.nn.xw_plus_b(self.h_drop,W,b)
				self.ypred = tf.nn.softmax(self.scores)
				self.preds = tf.argmax(self.scores,1)
			with tf.name_scope('loss'):
				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.in_y)
				self.loss = tf.reduce_mean(losses)+l2_lam*l2_loss
				self.d_loss = tf.reshape(tf.reduce_mean(self.loss),shape=[1])
		self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
		def d_optimizer(self, *args, **kwargs):
			return tf.train.AdamOptimizer(*args, **kwargs)
		d_opt = d_optimizer(lr)
		grad_and_vars = d_opt.compute_gradients(self.loss,self.params,aggregation_method=2)
		self.train_op = d_opt.apply_gradients(grad_and_vars)

class GenDataLoader(object):
	def __init__(self, b_size, sq_len, end=0):
		self.b_size = b_size
		self.sq_len = sq_len
		self.stream = []
		self.end = end
	def batches(self, fn):
		self.stream = []
		for ln in open(get_path(fn)).readlines():
			ln = map(int,ln.strip().split())
			self.stream.append(ln[:self.sq_len]+\
			[self.end]*max(0,self.sq_len-len(ln)))
		self.n_batch = int(len(self.stream)/self.b_size)
		self.stream = self.stream[:self.n_batch*self.b_size]
		self.seqs = np.split(np.array(self.stream),self.n_batch,0)
		self.pp = 0
	def next(self):
		ret, self.pp = self.seqs[self.pp], (self.pp+1)%self.n_batch
		return ret
	def reset(self):
		self.pp = 0

class DisDataLoader(object):
	def __init__(self, b_size, sq_len):
		self.b_size = b_size
		self.X = np.array([])
		self.Y = np.array([])
		self.sq_len = sq_len
	def batches(self, pos_fn, neg_fn):
		poss, negs = [], []
		for ln in open(get_path(pos_fn)).readlines():
			poss.append(map(int,ln.strip().split()))
		for ln in open(get_path(neg_fn)).readlines():
			negs.append(map(int,ln.strip().split()))
		self.X = np.array(poss+negs)
		self.Y = np.array([[0,1] for _ in poss]+[[1,0] for _ in negs])
		shuffle = np.random.permutation(np.arange(len(self.Y)))
		self.X, self.Y = self.X[shuffle], self.Y[shuffle]
		self.n_batch = int(len(self.Y)/self.b_size)
		self.X = self.X[:self.n_batch*self.b_size]
		self.Y = self.Y[:self.n_batch*self.b_size]
		self.X = np.split(np.array(self.X),self.n_batch,0)
		self.Y = np.split(np.array(self.Y),self.n_batch,0)
		self.pp = 0
	def next(self):
		ret, self.pp = (self.X[self.pp],self.Y[self.pp]), (self.pp+1)%self.n_batch
		return ret
	def reset(self):
		self.pp = 0

class Reward(object):
	def __init__(self, lstm, ur=.8):
		self.lstm = lstm
		self.v_size = self.lstm.v_size
		self.b_size = self.lstm.b_size
		self.e_dim = self.lstm.e_dim
		self.h_dim = self.lstm.h_dim
		self.sq_len = self.lstm.sq_len
		self.start = tf.identity(self.lstm.start)
		self.lr = self.lstm.lr
		self.ur = ur
		def recurrent_unit():
			self.Wi = tf.identity(self.lstm.Wi)
			self.Ui = tf.identity(self.lstm.Ui)
			self.bi = tf.identity(self.lstm.bi)
			self.Wf = tf.identity(self.lstm.Wf)
			self.Uf = tf.identity(self.lstm.Uf)
			self.bf = tf.identity(self.lstm.bf)
			self.Wg = tf.identity(self.lstm.Wg)
			self.Ug = tf.identity(self.lstm.Ug)
			self.bg = tf.identity(self.lstm.bg)
			self.Wc = tf.identity(self.lstm.Wc)
			self.Uc = tf.identity(self.lstm.Uc)
			self.bc = tf.identity(self.lstm.bc)
			def unit(x, h_mem):
				h_prev, c_prev = tf.unstack(h_mem)
				i = tf.sigmoid(tf.matmul(x,self.Wi)+tf.matmul(h_prev,self.Ui)+self.bi)
				f = tf.sigmoid(tf.matmul(x,self.Wf)+tf.matmul(h_prev,self.Uf)+self.bf)
				o = tf.sigmoid(tf.matmul(x,self.Wg)+tf.matmul(h_prev,self.Ug)+self.bg)
				c = f*c_prev+i*tf.nn.tanh(tf.matmul(x,self.Wc)+tf.matmul(h_prev,self.Uc)+self.bc)
				h = o*tf.nn.tanh(c)
				return tf.stack([h,c])
			return unit
		def output_unit():
			self.Wo = tf.identity(self.lstm.Wo)
			self.bo = tf.identity(self.lstm.bo)
			def unit(h_mem):
				h_prev, c_prev = tf.unstack(h_mem)
				logits = tf.matmul(h_prev,self.Wo)+self.bo
				return logits
			return unit
		self.g_embed = tf.identity(self.lstm.g_embed)
		self.g_runit = recurrent_unit()
		self.g_ounit = output_unit()
		# rollout
		self.given = tf.placeholder(tf.int32)
		self.h0 = tf.zeros([self.b_size, self.h_dim])
		self.h0 = tf.stack([self.h0,self.h0])
		gen_x = tensor_array_ops.TensorArray(dtype=tf.int32,size=self.sq_len,dynamic_size=False,infer_shape=True)
		self.x = tf.placeholder(tf.int32,shape=[self.b_size,self.sq_len])
		self.emb_x = tf.transpose(tf.nn.embedding_lookup(self.g_embed,self.x),perm=[1,0,2])
		tf_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32,size=self.sq_len) # teacher forcing
		tf_emb_x = tf_emb_x.unstack(self.emb_x)
		tf_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sq_len)
		tf_x = tf_x.unstack(tf.transpose(self.x,perm=[1,0]))
		def recurrent_1(i, x_t, h_p, given, gen_x):
			h_t = self.g_runit(x_t,h_p)
			x_n = tf_emb_x.read(i)
			gen_x = gen_x.write(i,tf_x.read(i))
			return i+1, x_n, h_t, given, gen_x
		def recurrent_2(i, x_t, h_p, given, gen_x):
			h_t = self.g_runit(x_t,h_p)
			o_t = self.g_ounit(h_t)
			prob = tf.log(tf.nn.softmax(o_t))
			next = tf.cast(tf.reshape(tf.multinomial(prob,1),[self.b_size]),tf.int32)
			x_n = tf.nn.embedding_lookup(self.g_embed,next)
			gen_x = gen_x.write(i,next)
			return i+1, x_n, h_t, given, gen_x
		i,x_t,h_p,given,self.gen_x = control_flow_ops.while_loop(
			cond=lambda i,_1,_2,given,_4:i<given, body=recurrent_1,
			loop_vars=(tf.constant(0,dtype=tf.int32),
				tf.nn.embedding_lookup(self.g_embed,self.start),self.h0,self.given,gen_x))
		_,_,_,_,self.gen_x = control_flow_ops.while_loop(
			cond=lambda i,_1,_2,_3,_4:i<self.sq_len, body=recurrent_2,
			loop_vars=(i,x_t,h_p,given,self.gen_x))
		self.gen_x = self.gen_x.stack()
		self.gen_x = tf.transpose(self.gen_x,perm=[1,0])
	def get_reward(self, sess, x, rollout, D_model):
		rewards = [0]*len(x[0])
		for i in xrange(rollout):
			for given in xrange(1,len(x[0])+1):
				samples = x if given == len(x[0]) else\
					sess.run(self.gen_x,{self.x:x,self.given:given}) 
				ypred = np.array(zip(*sess.run(D_model.ypred,{D_model.in_x:samples}))[1])
				rewards[given-1] += ypred
		return np.transpose(np.array(rewards))/(1.0*rollout)
	def update_recurrent_unit(self):
		self.Wi = self.ur*self.Wi+(1-self.ur)*tf.identity(self.lstm.Wi)
		self.Ui = self.ur*self.Ui+(1-self.ur)*tf.identity(self.lstm.Ui)
		self.bi = self.ur*self.bi+(1-self.ur)*tf.identity(self.lstm.bi)
		self.Wf = self.ur*self.Wf+(1-self.ur)*tf.identity(self.lstm.Wf)
		self.Uf = self.ur*self.Uf+(1-self.ur)*tf.identity(self.lstm.Uf)
		self.bf = self.ur*self.bf+(1-self.ur)*tf.identity(self.lstm.bf)
		self.Wg = self.ur*self.Wg+(1-self.ur)*tf.identity(self.lstm.Wg)
		self.Ug = self.ur*self.Ug+(1-self.ur)*tf.identity(self.lstm.Ug)
		self.bg = self.ur*self.bg+(1-self.ur)*tf.identity(self.lstm.bg)
		self.Wc = self.ur*self.Wc+(1-self.ur)*tf.identity(self.lstm.Wc)
		self.Uc = self.ur*self.Uc+(1-self.ur)*tf.identity(self.lstm.Uc)
		self.bc = self.ur*self.bc+(1-self.ur)*tf.identity(self.lstm.bc)
		def unit(x, h_mem):
			h_prev, c_prev = tf.unstack(h_mem)
			i = tf.sigmoid(tf.matmul(x,self.Wi)+tf.matmul(h_prev,self.Ui)+self.bi)
			f = tf.sigmoid(tf.matmul(x,self.Wf)+tf.matmul(h_prev,self.Uf)+self.bf)
			o = tf.sigmoid(tf.matmul(x,self.Wg)+tf.matmul(h_prev,self.Ug)+self.bg)
			c = f*c_prev+i*tf.nn.tanh(tf.matmul(x,self.Wc)+tf.matmul(h_prev,self.Uc)+self.bc)
			h = o*tf.nn.tanh(c)
			return tf.stack([h,c])
		return unit
	def update_output_unit(self):
		self.Wo = self.ur*self.Wo+(1-self.ur)*tf.identity(self.lstm.Wo)
		self.bo = self.ur*self.bo+(1-self.ur)*tf.identity(self.lstm.bo)
		def unit(h_mem):
			h_prev, c_prev = tf.unstack(h_mem)
			logits = tf.matmul(h_prev,self.Wo)+self.bo
			return logits
		return unit
	def update_params(self):
		self.g_embed = tf.identity(self.lstm.g_embed)
		self.g_runit = self.update_recurrent_unit()
		self.g_ounit = self.update_output_unit()

class SeqGan(Gan):

	def __init__(self):
		super(SeqGan,self).__init__()
		self.v_size = 20
		self.b_size = 64
		self.e_dim = 32
		self.h_dim = 32
		self.sq_len = 20
		self.gen_num = 128
		self.start = 0
		self.f_size = [2,3]
		self.f_num = [100,200]
		self.ora_fn = 'save/ora.txt'
		self.gen_fn = 'save/gen.txt'
		self.eva_fn = 'save/eva.txt'
		self.out_fn = 'save/out.txt'

	def init_train(self, fn):
		self.sq_len, self.v_size = text_proc(fn,self.eva_fn)
		self.G_model = Generator(v_size=self.v_size,b_size=self.b_size,\
								 e_dim=self.e_dim,h_dim=self.h_dim,\
								 sq_len=self.sq_len,start=self.start)
		self.D_model = Discriminator(v_size=self.v_size,sq_len=self.sq_len,e_dim=self.e_dim,\
									 f_size=self.f_size,f_num=self.f_num)
		self.G_dloader = GenDataLoader(b_size=self.b_size,sq_len=self.sq_len)
		self.D_dloader = DisDataLoader(b_size=self.b_size,sq_len=self.sq_len)
		words = list(tokenlize(fn)); wd2id, id2wd = gen_dict(words)
		with open(get_path(self.ora_fn),'w') as out:
			out.write(wds2ids(wd2id,words,self.sq_len))
		return wd2id, id2wd

	def init_metric(self, fn):
		from metrics import Bleu
		self.metrics.append(Bleu(test_text=self.out_fn,real_text=fn))

	def train_discriminator(self):
		gen_samples(self.sess,self.G_model,self.b_size,self.gen_num,self.gen_fn)
		self.D_dloader.batches(self.ora_fn,self.gen_fn)
		for _ in xrange(3):
			self.D_dloader.next()
			x,y = self.D_dloader.next()
			loss,_ = self.sess.run([self.D_model.d_loss,self.D_model.train_op],\
								   {self.D_model.in_x:x,self.D_model.in_y:y})

	def train(self, fn='../data/coco.txt'):
		wd2id, id2wd = self.init_train(fn)
		self.init_metric(fn)
		self.sess.run(tf.global_variables_initializer())
		gen_samples(self.sess,self.G_model,self.b_size,self.gen_num,self.gen_fn)
		self.G_dloader.batches(self.ora_fn)
		print 'start pre-train generator:'
		for epoch in xrange(self.pre_epochs):
			loss = pre_train(self.sess,self.G_model,self.G_dloader)
			print 'epoch:',epoch,loss
		self.evaluate()
		print 'start pre-train discriminator:'
		for epoch in xrange(self.pre_epochs):
			print 'epoch:',epoch
			self.train_discriminator()
		print 'adversarial training:'
		self.reward = Reward(self.G_model)
		for epoch in xrange(self.adv_epochs):
			samples = self.G_model.generate(self.sess)
			rewards = self.reward.get_reward(self.sess,samples,16,self.D_model)
			feed = {self.G_model.x:samples,self.G_model.r:rewards}
			loss,_ = self.sess.run([self.G_model.g_loss,self.G_model.g_updates],feed_dict=feed)	
			print 'epoch:',epoch,loss
			if epoch%5 == 0 or epoch == self.adv_epochs-1:
				gen_samples(self.sess,self.G_model,self.b_size,self.gen_num,self.gen_fn)
				with open(get_path(self.out_fn),'w') as out: # gen out file
					out.write(ids2wds(id2wd,tokenlize(self.gen_fn)))
				self.evaluate()
			self.reward.update_params()
			for _ in xrange(15):
				self.train_discriminator()

if __name__ == '__main__':
	pass
