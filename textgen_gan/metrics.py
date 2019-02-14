# -*- coding: utf-8 -*-

import nltk
from utils import *
from multiprocessing import Pool, cpu_count
from nltk.translate.bleu_score import SmoothingFunction

def calc_bleu(reference, hypothesis, weight):
	return nltk.translate.bleu_score.sentence_bleu(\
		reference,hypothesis,weight,smoothing_function=SmoothingFunction().method1)
	
class Bleu(object):
	def __init__(self, test_text='', real_text='', gram=3):
		self.name = 'Bleu'
		self.test_data = test_text
		self.real_data = real_text
		self.gram = gram
		self.reference = []
	def get_score(self):
		if not self.reference:
			self.reference = [
				nltk.word_tokenize(line.decode('utf-8').strip().lower())\
				for line in open(get_path(self.real_data)).readlines()]
		return self.get_bleu()
	def get_bleu(self):
		result = []; pool = Pool(cpu_count())
		weight = tuple([1./self.gram]*self.gram)
		with open(get_path(self.test_data)) as test_data:
			for hypothesis in test_data:
				hypothesis = nltk.word_tokenize(hypothesis.decode('utf-8').strip().lower())
				result.append(pool.apply_async(calc_bleu,args=(self.reference,hypothesis,weight)))
		score = [i.get() for i in result]
		pool.close(); pool.join()
		return sum(score)/len(score)

if __name__ == '__main__':
	pass

