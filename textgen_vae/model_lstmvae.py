# -*- coding: utf-8 -*-

import os, nltk, random, numpy as np
from nltk.tokenize import word_tokenize
from keras import backend as K
from keras.models import Model
from keras.layers import Input, GRU, LSTM, RepeatVector, TimeDistributed
from keras.layers.core import Dense, Lambda
from keras import objectives

get_path = lambda fn:os.path.join(os.path.abspath(os.path.dirname(__file__)),fn)

def get_data(fn, samples):
	texts = []; words = set([u'\t'])
	for line in open(get_path(fn)).readlines()[:samples]:
		line = line.decode('utf-8').split(u'\t')[0].strip()
		line = nltk.word_tokenize(line.lower())+[u'<end>']
		texts.append(line); words |= set(line)
	words = sorted(list(words))
	v_size, sq_len = len(words), max(map(len,texts))+1
	wd2id = {w:i for i,w in enumerate(words)}
	id2wd = {i:w for w,i in wd2id.iteritems()}
	enc_x = np.zeros((len(texts),sq_len,v_size),dtype='float32')
	dec_x = np.zeros((len(texts),sq_len,v_size),dtype='float32')
	for i,text in enumerate(texts):
		dec_x[i,0,wd2id[u'\t']] = 1.0
		for t,w in enumerate(text):
			enc_x[i,t,wd2id[w]] = 1.0
			dec_x[i,t+1,wd2id[w]] = 1.0
	return sq_len, v_size, texts, wd2id, id2wd, enc_x, dec_x

def lstm_vae(sq_len,x_dim,h_dim,z_dim,b_size):
	x = Input(shape=(None,x_dim))
	h = LSTM(h_dim)(x)
	z_m,z_lv = Dense(z_dim)(h),Dense(z_dim)(h)
	sampling = lambda args: args[0]+K.exp(args[1]/2)*K.random_normal(shape=(b_size,z_dim),mean=0.,stddev=1.)
	z = Lambda(sampling,output_shape=(z_dim,))([z_m,z_lv])
	reweight = Dense(h_dim,activation='linear')
	z = reweight(z)
	dx = Input(shape=(None,x_dim)) # teacher forcing
	decoder_h = LSTM(h_dim,return_sequences=True,return_state=True)
	decoder_o = TimeDistributed(Dense(x_dim,activation='softmax'))
	dh,_,_ = decoder_h(dx,initial_state=[z,z])
	do = decoder_o(dh)
	vae = Model([x,dx],do)
	enc = Model(x,[z_m,z_lv])
	ds = Input(shape=(z_dim,))
	_z = reweight(ds)
	_dh,_rh,_rc = decoder_h(dx,initial_state=[_z,_z])
	_do = decoder_o(_dh)
	gen = Model([dx,ds],[_do,_rh,_rc])
	drh = Input(shape=(h_dim,))
	drc = Input(shape=(h_dim,))
	__dh,__rh,__rc = decoder_h(dx,initial_state=[drh,drc])
	__do = decoder_o(__dh)
	step = Model([dx,drh,drc],[__do,__rh,__rc])
	loss = lambda x,do:objectives.categorical_crossentropy(x,do)-0.5*K.mean(1+z_lv-K.square(z_m)-K.exp(z_lv))
	vae.compile(optimizer='adam',loss=loss)
	vae.summary()
	return vae, enc, gen, step

def decode(s, gen1, step, x_dim, sq_len, wd2id, id2wd):
	tseq = np.zeros((1,1,x_dim))
	tseq[0,0,wd2id[u'\t']] = 1.0
	h,c = None,None
	for i in xrange(sq_len):
		o,h,c = gen1.predict([tseq,s]) if i == 0 else\
				step.predict([tseq,h,c])
		word = id2wd[np.argmax(o[0,-1,:])]; yield word
		if word == u'<end>': break
		tseq = np.zeros((1,1,x_dim))
		tseq[0,0,wd2id[word]] = 1.0

if __name__ == '__main__':
	sq_len,x_dim,texts,wd2id,id2wd,enc_x,dec_x = get_data('../data/fra.txt',3000)
	h_dim,z_dim,b_size,epochs = 256,256,1,30
	vae,enc,gen1,step = lstm_vae(sq_len,x_dim,h_dim,z_dim,b_size)
	vae.fit([enc_x,dec_x],enc_x,epochs=epochs)
	for _ in range(5):
		id1 = np.random.randint(0,len(texts)-1)
		id2 = np.random.randint(0,len(texts)-1)
		m1,v1 = enc.predict(np.array([enc_x[id1]]))
		m2,v2 = enc.predict(np.array([enc_x[id2]]))
		seq1 = m1+v1*np.random.normal(size=(z_dim,))
		seq2 = m2+v2*np.random.normal(size=(z_dim,))
		print '==\t',' '.join([id2wd[i] for i in np.argmax(enc_x[id1],axis=1)]).strip(),'=='
		for v in np.linspace(0,1,7):
			print round(1-v,2),u' '.join(list(decode(v*seq2+(1-v)*seq1,\
					gen1,step,x_dim,sq_len,wd2id,id2wd)))
		print '==\t',' '.join([id2wd[i] for i in np.argmax(enc_x[id2],axis=1)]).strip(),'=='
