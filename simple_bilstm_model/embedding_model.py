#coding:utf-8
# 预训练词向量
import gensim,os
import codecs

class MySentences():
	def __init__(self,dirname_list):
		self.dirname_list=dirname_list

	def __iter__(self):

		for dirname in self.dirname_list:
			for fname in os.listdir(dirname):
				for line in codecs.open(dirname+os.sep+fname,'r','utf-8'):
					pieces=line.strip().replace(' ','')
					words=[w for w in pieces]
					yield words

corpus_path='corpus'
sentences=MySentences([corpus_path+os.sep+fname for fname in os.listdir(corpus_path)])

model=gensim.models.Word2Vec(sentences,size=100,window=5,min_count=1,iter=100,workers=4)
model.save('model_conll_law.m')
model.wv.save_word2vec_format('model_conll_law.txt',binary=False)
