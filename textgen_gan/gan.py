# -*- coding: utf-8 -*-

import tensorflow as tf

def init_sess():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	return sess

class Gan(object):

	def __init__(self):
		self.G_model = None
		self.D_model = None
		self.G_dloader = None
		self.D_dloader = None
		self.sess = init_sess()
		self.epoch = 0
		self.pre_epochs = 80
		self.adv_epochs = 100
		self.metrics = []

	def evaluate(self):
		for metric in self.metrics:
			print metric.name, metric.get_score()

	def train(self):
		pass

if __name__ == '__main__':
	pass
