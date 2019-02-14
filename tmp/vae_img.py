# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.losses import binary_crossentropy
from keras.datasets import mnist

x_dim, h_dim, z_dim = 28*28, 256, 2
b_size, epochs = 100, 10

x = Input(batch_shape=(b_size,x_dim))
h = Dense(h_dim,activation='relu')(x)
z_m,z_lv = Dense(z_dim)(h), Dense(z_dim)(h)
sampling = lambda args: args[0]+K.exp(args[1]/2)*K.random_normal(shape=(b_size,z_dim),mean=0.,stddev=1.)
z = Lambda(sampling,output_shape=(z_dim,))([z_m,z_lv])
decoder_h = Dense(z_dim,activation='relu')
decoder_m = Dense(x_dim,activation='sigmoid')
dx = decoder_m(decoder_h(z))
loss = lambda x,dx:x_dim*binary_crossentropy(x,dx)-0.5*K.sum(1+z_lv-K.square(z_m)-K.exp(z_lv),axis=-1)
model = Model(x,dx)
model.compile(optimizer='rmsprop',loss=loss)
 
(x_tr,y_tr),(x_te,y_te) = mnist.load_data()
x_tr = (x_tr.astype('float32')/255.).reshape((len(x_tr),np.prod(x_tr.shape[1:])))
x_te = (x_te.astype('float32')/255.).reshape((len(x_te),np.prod(x_te.shape[1:])))
model.fit(x_tr,x_tr,nb_epoch=epochs,batch_size=b_size,validation_data=(x_te,x_te))

# plot
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# classes in latent space
encoder = Model(x,z_m)
x_te = encoder.predict(x_te,batch_size=b_size)
plt.figure(figsize=(6,6))
plt.scatter(x_te[:,0],x_te[:,1],c=y_te)
plt.colorbar()
plt.savefig('classes.png')
# manifold of the classes
d = Input(shape=(z_dim,))
dm = decoder_m(decoder_h(d))
generator = Model(d,dm)
n = 15; img_size = 28
figure = np.zeros((img_size*n,img_size*n))
grid_x = norm.ppf(np.linspace(0.05,0.95,n))
grid_y = norm.ppf(np.linspace(0.05,0.95,n))
for i,yi in enumerate(grid_x):
	for j,xj in enumerate(grid_y):
		z_sample = np.array([[xj,yi]])
		x_decoded = generator.predict(z_sample)
		img = x_decoded[0].reshape(img_size, img_size)
		figure[i*img_size:(i+1)*img_size,\
			   j*img_size:(j+1)*img_size] = img
plt.figure(figsize=(10,10))
plt.imshow(figure,cmap='Greys_r')
plt.savefig('manifold.png')
