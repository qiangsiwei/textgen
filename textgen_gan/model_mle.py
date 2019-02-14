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
	def pretrain(self, sess, x):
		return sess.run([self.pretrain_updates,self.pretrain_loss],feed_dict={self.x:x})
	def generate(self, sess):
		return sess.run(self.gen_x)

class DataLoader(object):
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

class MLE(Gan):

	def __init__(self):
		super(MLE,self).__init__()
		self.v_size = 20
		self.b_size = 64
		self.e_dim = 32
		self.h_dim = 32
		self.sq_len = 20
		self.gen_num = 128
		self.start = 0
		self.ora_fn = 'save/ora.txt'
		self.gen_fn = 'save/gen.txt'
		self.eva_fn = 'save/eva.txt'
		self.out_fn = 'save/out.txt'

	def init_train(self, fn):
		self.sq_len, self.v_size = text_proc(fn,self.eva_fn)
		self.G_model = Generator(v_size=self.v_size,b_size=self.b_size,\
								 e_dim=self.e_dim,h_dim=self.h_dim,\
								 sq_len=self.sq_len,start=self.start)
		self.G_dloader = DataLoader(b_size=self.b_size,sq_len=self.sq_len)
		words = list(tokenlize(fn)); wd2id, id2wd = gen_dict(words)
		with open(get_path(self.ora_fn),'w') as out:
			out.write(wds2ids(wd2id,words,self.sq_len))
		return wd2id, id2wd

	def train(self, fn='../data/coco.txt'):
		wd2id, id2wd = self.init_train(fn)
		self.sess.run(tf.global_variables_initializer())
		gen_samples(self.sess,self.G_model,self.b_size,self.gen_num,self.gen_fn)
		self.G_dloader.batches(self.ora_fn)
		print 'start pre-train generator:'
		for epoch in xrange(self.pre_epochs):
			loss = pre_train(self.sess,self.G_model,self.G_dloader)
			print 'epoch:',self.epoch,loss
			self.epoch += 1
			if epoch%5 == 0:
				gen_samples(self.sess,self.G_model,self.b_size,self.gen_num,self.gen_fn)
				with open(get_path(self.out_fn),'w') as out: # gen out file
					out.write(ids2wds(id2wd,tokenlize(self.gen_fn)))
				self.evaluate()

if __name__ == '__main__':
	pass
