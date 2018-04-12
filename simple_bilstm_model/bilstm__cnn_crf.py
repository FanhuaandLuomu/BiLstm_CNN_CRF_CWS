#coding:utf-8
# py3.5+tensorflow-1.0.1+keras-2.0.6
# seq2seq bilstm+cnn+crf

# pip install keras==2.0.6
# keras_contrib==2.0.8 pip install git+https://www.github.com/keras-team/keras-contrib.git

import os,re
import codecs 
import pickle
import time

import bottle
import jieba
jieba.initialize()

import gc
import numpy as np
np.random.seed(1111)

import gensim
import random

import keras
# keras.backend.clear_session() 

from keras.layers import *
from keras.models import *
from keras_contrib.layers import CRF

from keras import backend as K

from keras.utils import plot_model
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

from keras.models import model_from_json

# input:
# maxlen  char_value_dict_len  class_label_count
def Bilstm_CNN_Crf(maxlen,char_value_dict_len,class_label_count,embedding_weights=None,is_train=True):
	word_input=Input(shape=(maxlen,),dtype='int32',name='word_input')
	if is_train:
		word_emb=Embedding(char_value_dict_len+2,output_dim=100,\
					input_length=maxlen,weights=[embedding_weights],\
					name='word_emb')(word_input)
	else:
		word_emb=Embedding(char_value_dict_len+2,output_dim=100,\
					input_length=maxlen,\
					name='word_emb')(word_input)
	
	# bilstm
	bilstm=Bidirectional(LSTM(64,return_sequences=True))(word_emb)
	bilstm_d=Dropout(0.1)(bilstm)

	# cnn
	half_window_size=2
	padding_layer=ZeroPadding1D(padding=half_window_size)(word_emb)
	conv=Conv1D(nb_filter=50,filter_length=2*half_window_size+1,\
			padding='valid')(padding_layer)
	conv_d=Dropout(0.1)(conv)
	dense_conv=TimeDistributed(Dense(50))(conv_d)

	# merge
	rnn_cnn_merge=merge([bilstm_d,dense_conv],mode='concat',concat_axis=2)
	dense=TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)

	# crf
	crf=CRF(class_label_count,sparse_target=False)
	crf_output=crf(dense)

	# build model
	model=Model(input=[word_input],output=[crf_output])

	model.compile(loss=crf.loss_function,optimizer='adam',metrics=[crf.accuracy])

	# model.summary()

	return model

class Documents():
	def __init__(self,chars,labels,index):
		self.chars=chars
		self.labels=labels
		self.index=index

# 读取数据
def create_documents(file_name):
	documents=[]
	chars,labels=[],[]

	with codecs.open(file_name,'r','utf-8') as f:
		index=0
		for line in f:

			line=line.strip()
			
			if len(line)==0:
				if len(chars)!=0:
					documents.append(Documents(chars,labels,index))
					chars=[]
					labels=[]
				index+=1

			else:
				pieces=line.strip().split()
				chars.append(pieces[0])
				labels.append(pieces[1])

				if pieces[0] in ['。','，','；']:
					documents.append(Documents(chars,labels,index))
					chars=[]
					labels=[]

		if len(chars)!=0:
				documents.append(Documents(chars,labels,index))
				chars,labels=[],[]
	return documents


# 生成词典
def get_lexicon(all_documents):
	chars={}
	for doc in all_documents:
		for char in doc.chars:
			chars[char]=chars.get(char,0)+1

	sorted_chars=sorted(chars.items(),key=lambda x:x[1],reverse=True)

	# 下标从1开始 0用来补长
	lexicon=dict([(item[0],index+1) for index,item in enumerate(sorted_chars)])
	lexicon_reverse=dict([(index+1,item[0]) for index,item in enumerate(sorted_chars)])
	return lexicon,lexicon_reverse

def create_embedding(embedding_model,embedding_size,lexicon_reverse):
	embedding_weights=np.zeros((len(lexicon_reverse)+2,embedding_size))

	for i in range(len(lexicon_reverse)):
		embedding_weights[i+1]=embedding_model[lexicon_reverse[i+1]]

	embedding_weights[-1]=np.random.uniform(-1,1,embedding_size)

	return embedding_weights


def create_matrix(documents,lexicon,label_2_index):
	data_list=[]
	label_list=[]
	index_list=[]
	for doc in documents:
		data_tmp=[]
		label_tmp=[]

		for char,label in zip(doc.chars,doc.labels):
			data_tmp.append(lexicon[char])
			label_tmp.append(label_2_index[label])

		data_list.append(data_tmp)
		label_list.append(label_tmp)
		index_list.append(doc.index)

	return data_list,label_list,index_list


def padding_sentences(data_list,label_list,max_len):
	padding_data_list=sequence.pad_sequences(data_list,maxlen=max_len)
	padding_label_list=[]
	for item in label_list:
		padding_label_list.append([0]*(max_len-len(item))+item)

	return padding_data_list,np.array(padding_label_list)


def process_data(s_file_list,t_file):
	ft=codecs.open(t_file,'w','utf-8')
	k=0
	for s_file in s_file_list:
		with codecs.open(s_file,'r','utf-8') as fs:
			lines=fs.readlines()
			# print(len(lines))
			for line in lines:
				word_list=line.strip().split()
				for word in word_list:
					if len(word)==1:
						ft.write(word+'\tS\n')
					else:
						ft.write(word[0]+'\tB\n')
						for w in word[1:-1]:
							ft.write(w+'\tM\n')
						ft.write(word[-1]+'\tE\n')
				ft.write('\n')
	ft.close()

# 训练模型 保存weights
def process_train(corpus_path,nb_epoch,base_model_weight=None):
	# 训练语料
	raw_train_file=[corpus_path+os.sep+type_path+os.sep+type_file \
				for type_path in os.listdir(corpus_path) \
				for type_file in os.listdir(corpus_path+os.sep+type_path)]


	process_data(raw_train_file,'train.data')

	train_documents=create_documents('train.data')

	print(len(train_documents))

	# for doc in train_documents[-20:]:
	# 	print(doc.index,doc.chars,doc.labels)

	# 生成词典
	lexicon,lexicon_reverse=get_lexicon(train_documents)
	print(len(lexicon),len(lexicon_reverse))

	embedding_model=gensim.models.Word2Vec.load(r'model_conll_law.m')
	embedding_size=embedding_model.vector_size
	print(embedding_size)

	# 预训练词向量
	embedding_weights=create_embedding(embedding_model,embedding_size,lexicon_reverse)
	print(embedding_weights.shape)
	# print(embedding_weights[-1])

	# print(lexicon_reverse[4351])
	# print(embedding_weights[-2])

	# 0 为padding的label
	label_2_index={'Pad':0,'B':1,'M':2,'E':3,'S':4,'Unk':5}
	index_2_label={0:'Pad',1:'B',2:'M',3:'E',4:'S',5:'Unk'}

	train_data_list,train_label_list,train_index_list=create_matrix(train_documents,lexicon,label_2_index)
	print(len(train_data_list),len(train_label_list),len(train_label_list))
	print(train_data_list[0])
	print(train_label_list[0])

	max_len=max(map(len,train_data_list))
	print('maxlen:',max_len)

	train_data_array,train_label_list_padding=padding_sentences(train_data_list,train_label_list,max_len)
	print(train_data_array.shape)
	print(train_data_array[0])

	train_label_array=np_utils.to_categorical(train_label_list_padding,len(label_2_index)).\
					reshape((len(train_label_list_padding),len(train_label_list_padding[0]),-1))
	print(train_label_array.shape)
	print(train_label_array[0])

	# model
	model=Bilstm_CNN_Crf(max_len,len(lexicon),len(label_2_index),embedding_weights)
	print(model.input_shape)
	print(model.output_shape)

	plot_model(model, to_file='bilstm_cnn_crf_model.png',show_shapes=True,show_layer_names=True)

	if base_model_weight!=None and os.path.exists(base_model_weight)==True:
		model.load_weights(base_model_weight)

	hist=model.fit(train_data_array,train_label_array,batch_size=256,epochs=nb_epoch,verbose=1)

	# model.load_weights('best_val_model.hdf5')
	

	'''
	test_y_pred=model.predict(train_data_array,batch_size=512,verbose=1)
	pred_label=np.argmax(test_y_pred,axis=2)
	print(pred_label[0])
	
	'''
	score=model.evaluate(train_data_array,train_label_array,batch_size=512)
	print(score)
	

	# save model
	model.save_weights('train_model.hdf5')

	# save lexicon
	pickle.dump([lexicon,lexicon_reverse,max_len,index_2_label],open('lexicon.pkl','wb'))


#===========Test================


# input:text
def process_test(text,lexicon,max_len,model):
	test_list=[]
	for c in text:
		test_list.append(lexicon.get(c,len(lexicon)+1))

	padding_test_array=sequence.pad_sequences([test_list],maxlen=max_len)
	# print(padding_test_array.shape)

	test_y_pred=model.predict(padding_test_array,verbose=1)
	pred_label=np.argmax(test_y_pred,axis=2)
	# print(pred_label[0])

	return pred_label[0],padding_test_array[0]


def create_pred_text(text,pred_label):

	start_index=len(pred_label)-len(text)
	pred_label=pred_label[start_index:]

	pred_text=''
	for p,t in zip(pred_label,text):
		if p in [0,3,4,5]:
			pred_text+=(t+' ')
		else:
			pred_text+=t

	return pred_text,pred_label

# lexicon,lexicon_reverse,max_len,index_2_label=pickle.load(open('lexicon.pkl','rb'))
# # model
# model=Bilstm_CNN_Crf(max_len,len(lexicon),len(index_2_label),is_train=False)
# model.load_weights('train_model.hdf5')

# 句子长度太长会截断
def word_seg(text):
	# train  需要训练就取消这部分注释☆☆☆☆☆☆☆
	# corpus_path='corpus'
	# nb_epoch=5
	# process_train(corpus_path,nb_epoch)
	#=========Test===========

	# raw_len=len(text)
	pred_label,padding_test_array=process_test(text,lexicon,max_len,model)

	pred_text,pred_label=create_pred_text(text,pred_label)

	# print(pred_text)
	# print(pred_label)

	return pred_text,pred_label

import math

def word_seg_by_sentences(text,model,lexicon,max_len):
	'''
	# 长度1001切分 不好 如：中国人|寿
	count=math.ceil(len(text)/100)
	text_list=[]
	text_list2=[]
	for i in range(count):
		# text_list.append(text[i*100:(i+1)*100])
		tmp=text[i*100:(i+1)*100]
		tmp2=[]
		for c in tmp:
			tmp2.append(lexicon.get(c,len(lexicon)+1))
		text_list.append(tmp)
		text_list2.append(tmp2)
	'''

	text_list=[]
	text_list2=[]
	i=0
	for j in range(len(text)):
		if text[j] in ['，','。','！','；','？'] or i+100<=j:
			tmp=text[i:j+1]
			i=j+1

			tmp2=[]
			for c in tmp:
				tmp2.append(lexicon.get(c,len(lexicon)+1))
			text_list.append(tmp)
			text_list2.append(tmp2)

	if i!=j+1:
		tmp=text[i:j+1]
		tmp2=[]
		for c in tmp:
			tmp2.append(lexicon.get(c,len(lexicon)+1))
		text_list.append(tmp)
		text_list2.append(tmp2)


	padding_test_array=sequence.pad_sequences(text_list2,maxlen=max_len)
	
	test_y_pred=model.predict(padding_test_array,verbose=1)
	pred_label_list=np.argmax(test_y_pred,axis=2)


	pred_text_all=''
	pred_label_all=[]
	for text,label in zip(text_list,pred_label_list):
		pred_text,pred_label=create_pred_text(text,label)
		pred_text_all+=pred_text
		pred_label_all.extend(pred_label)

	return pred_text_all,pred_label_all


def fenci_by_file(source_path,target_path,model,lexicon,max_len):

	if not os.path.exists(target_path):
		os.mkdir(target_path)


	for filename in os.listdir(source_path):
		lines=codecs.open(source_path+os.sep+filename,'r','utf-8').readlines()
		
		f=codecs.open(target_path+os.sep+filename,'w','utf-8')
		for line in lines:
			line=line.strip()

			# splitText=' '.join(jieba.cut(line))

			splitText,_=word_seg_by_sentences(line,model,lexicon,max_len)

			f.write(splitText+'\n')
		f.close()
	print('fenci success!')

def main():

	## note
	# 把你的语料放到corpus文件夹下
	# 1. python embedding_model.py  -> model_conll_law.m  生成词向量文件
	# 2. python bilstm_cnn_crf.py    // is_train==1
	# 会得到 train_model.hdf5  lexicon.pkl
	# 3. 可以在之前的基础上train_model.hdf5，继续训练
	# 4. 训练完成，测试  is_train==0
	# python bilstm_cnn_crf.py  按句测试或按文件测试

	# my_weights 中存放的是我的权值 


	is_train=1  # 1/0
	
	if is_train==1:
		# train  ☆☆☆☆☆☆☆
		# 训练语料路径
		corpus_path='corpus'
		# 初始化模型参数  可在之前的基础上训练
		base_model_weight='train_model.hdf5'
		nb_epoch=1   # 迭代轮数
		process_train(corpus_path,nb_epoch,base_model_weight)
	

	##############################################

	lexicon,lexicon_reverse,max_len,index_2_label=pickle.load(open('lexicon.pkl','rb'))
	# model
	model=Bilstm_CNN_Crf(max_len,len(lexicon),len(index_2_label),is_train=False)
	model.load_weights('train_model.hdf5')

	# 长句子测试   按标点切分后测试
	text=''
	for i in range(10):
		text+='南京市长莅临指导，大家热烈欢迎。公交车中将禁止吃东西！'
	splitText,predLabel=word_seg_by_sentences(text,model,lexicon,max_len)
	print(splitText)
	
	fenci_by_file('test_documents/test_1','test_documents/test_1_mine',model,lexicon,max_len)
	

if __name__ == '__main__':

	main()

