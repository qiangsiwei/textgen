# -*- coding: utf-8 -*-

import os, nltk, numpy as np

get_path = lambda fn:os.path.join(os.path.abspath(os.path.dirname(__file__)),fn)

def ids2wds(dc, ids):
	eof = len(dc)
	return u'\n'.join([u' '.join(\
		[str(dc[int(_id)]) for _id in _ids\
		if int(_id)!=eof]) for _ids in ids])

def wds2ids(dc, wds, sq_len):
	eof = len(dc)
	return u'\n'.join([u' '.join(\
		[str(dc[_wd]) for _wd in _wds]+\
		[str(eof)]*(sq_len-len(_wds))) for _wds in wds])

def gen_dict(words):
	words = set([w for wds in words for w in wds])
	wd2id = {w:i+1 for i,w in enumerate(words)}
	id2wd = {i:w for w,i in wd2id.iteritems()}
	return wd2id, id2wd

def tokenlize(fn):
	for line in open(get_path(fn)).readlines():
		yield nltk.word_tokenize(line.decode('utf-8').strip().lower())

def text_proc(fn, out_fn):
	words = list(tokenlize(fn))
	sq_len = max(map(len,words))
	wd2id, id2wd = gen_dict(words)
	with open(get_path(out_fn),'w') as out:
		out.write(wds2ids(wd2id,words,sq_len))
	return sq_len, len(wd2id)+1 

def pre_train(sess, model, dloader):
    g_losses = []; dloader.reset()
    for it in xrange(dloader.n_batch):
        g_losses.append(model.pretrain(sess,dloader.next())[1])
    return np.mean(g_losses)

def gen_samples(sess, model, b_size, gen_num, out_fn=None):
	samples = []
	for _ in xrange(int(gen_num/b_size)):
		samples.extend(model.generate(sess))
	out_str = u'\n'.join([u' '.join(map(str,s)) for s in samples])
	if not out_fn: return out_str
	with open(get_path(out_fn),'w') as out:
		out.write(out_str)
	return np.array(samples)

if __name__ == '__main__':
	text_proc('../data/coco.txt')
