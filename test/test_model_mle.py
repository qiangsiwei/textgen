# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from textgen_gan.model_mle import MLE

if __name__ == '__main__':
	MLE().train(fn='../data/coco.txt')
