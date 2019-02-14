# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from textgen_gan.model_seqgan import SeqGan

if __name__ == '__main__':
	SeqGan().train(fn='../data/coco.txt')
