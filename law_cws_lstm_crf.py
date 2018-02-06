#coding:utf-8
# py3.5+tensorflow-1.0.1+keras-2.0.6
# seq2seq bilstm+cnn+crf
import os
import codecs 

import gc
import numpy as np
# np.random.seed(1111)

import gensim
import random

import keras
from keras.layers import *
from keras.models import *
from keras_contrib.layers import CRF

from keras import backend as K

from keras.utils import plot_model
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

# input:
# maxlen  char_value_dict_len  class_label_count
def Bilstm_CNN_Crf(maxlen,char_value_dict_len,class_label_count,embedding_weights):
	word_input=Input(shape=(maxlen,),dtype='int32',name='word_input')
	word_emb=Embedding(char_value_dict_len+1,output_dim=100,\
				input_length=maxlen,weights=[embedding_weights],\
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
	embedding_weights=np.zeros((len(lexicon_reverse)+1,embedding_size))
	for i in range(len(lexicon_reverse)):
		embedding_weights[i+1]=embedding_model[lexicon_reverse[i+1]]
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

# 可视化
def plot_acc(hist):
	import matplotlib.pyplot as plt
	plt.plot(range(len(hist.history['acc'])),hist.history['acc'],marker='o',label='acc')
	plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],marker='*',label='val_acc')
	plt.legend()
	plt.xlabel('iters')
	plt.ylabel('acc_')
	plt.title('Acc & val_Acc')
	plt.show()


def create_pred_text(lexicon_reverse,test_data_array,pred_label,test_label_list_padding,test_index_list):
	real_text_list=[]
	pred_text_list=[]
	real_label_list=[]
	pred_label_list=[]

	real_text=''
	pred_text=''

	non_pad_real=[]
	non_pad_pred=[]
	non_pad_text=[]

	sindex=0

	for pred,real,index,text in zip(pred_label,test_label_list_padding,test_index_list,test_data_array):
		start_index=np.argwhere(real>0)[0][0]
		# print(start_index)
	
		if index!=sindex:

			real_text_list.append(real_text)
			pred_text_list.append(pred_text)

			real_label_list.append(non_pad_real)
			pred_label_list.append(non_pad_pred)

			real_text=''
			pred_text=''

			non_pad_real=[]
			non_pad_pred=[]
			non_pad_text=[]

		for r,p,t in zip(real[start_index:],pred[start_index:],text[start_index:]):

			# E
			if r in [0,3,4]:
				real_text+=(lexicon_reverse[t]+' ')
			else:
				real_text+=lexicon_reverse[t]

			# E
			if p in [0,3,4]:
				pred_text+=(lexicon_reverse[t]+' ')
			else:
				pred_text+=lexicon_reverse[t]

		non_pad_real+=list(real[start_index:])
		non_pad_pred+=list(pred[start_index:])
		non_pad_text+=list(text[start_index:])

		sindex=index

	if pred_text!='':

		real_text_list.append(real_text)
		pred_text_list.append(pred_text)

		real_label_list.append(non_pad_real)
		pred_label_list.append(non_pad_pred)

		real_text=''
		pred_text=''

		non_pad_real=[]
		non_pad_pred=[]
		non_pad_text=[]

	return real_text_list,pred_text_list,real_label_list,pred_label_list

def write_2_file(real_text_list,pred_text_list):
	f=codecs.open('real_text.txt','w','utf-8')
	f.write('\n'.join(real_text_list))
	f.close()

	f=codecs.open('pred_text.txt','w','utf-8')
	f.write('\n'.join(pred_text_list))
	f.close()

import get_train_test
import create_format_data
import score

# 打乱文件顺序，抽取train和test
# 语料文件夹  
# 随机数seed，默认1111  
# k:train-test 划分比例  默认0.8
def step_1(spath,t_path,seed=1111,k=0.8):
	# spath='biaozhu_1_100'
	filenames=os.listdir(spath)

	# seed=2222
	random.seed(seed)
	random.shuffle(filenames)

	train_files=filenames[:int(k*len(filenames))]
	test_files=filenames[int(k*len(filenames)):]

	# t_path='corpus'

	get_train_test.create_file(spath,train_files,t_path+os.sep+'train_p%s_%d.utf8' %(k,seed))
	get_train_test.create_file(spath,test_files,t_path+os.sep+'test_p%.1f_%d.utf8' %(1.0-k,seed))

# 标注样本
def step_2(t_path,conll_path,seed,k):
	# seed=2222
	raw_train_file=[t_path+os.sep+'train_p%s_%d.utf8' %(k,seed)]
	# 添加conll2012分词训练语料
	raw_train_file+=[conll_path+os.sep+fname for fname in os.listdir(conll_path)]

	raw_test_file=[t_path+os.sep+'test_p%.1f_%d.utf8' %(1.0-k,seed)]

	create_format_data.process(raw_train_file,'train_%d.data' %seed)
	create_format_data.process(raw_test_file,'test_%d.data' %seed)


def process(spath,t_path,conll_path,text_seed,k,prf_file):
	step_1(spath,t_path,text_seed,k)

	# step_2
	# 法律文档*k+conll2012 语料训练
	step_2(t_path,conll_path,text_seed,k)

	# step_3


	# text_seed=2222
	train_file='train_%d.data' %(text_seed)
	test_file='test_%d.data' %(text_seed)
	train_documents=create_documents(train_file)
	print(len(train_documents))
	# print(train_documents[1].chars)
	# print(train_documents[1].labels)
	for doc in train_documents[:20]:
		print(doc.index,doc.chars,doc.labels)

	test_documents=create_documents(test_file)
	print(len(test_documents))

	# 生成词典
	lexicon,lexicon_reverse=get_lexicon(train_documents+test_documents)
	print(len(lexicon),len(lexicon_reverse))

	embedding_model=gensim.models.Word2Vec.load(r'model_conll_law.m')
	embedding_size=embedding_model.vector_size
	print(embedding_size)

	# 预训练词向量
	embedding_weights=create_embedding(embedding_model,embedding_size,lexicon_reverse)

	print(embedding_weights.shape)

	print(lexicon_reverse[1])
	print(embedding_weights[1])

	# 0 为padding的label
	label_2_index={'B':1,'M':2,'E':3,'S':4}
	index_2_label={0:'Pad',1:'B',2:'M',3:'E',4:'S'}

	train_data_list,train_label_list,train_index_list=create_matrix(train_documents,lexicon,label_2_index)
	test_data_list,test_label_list,test_index_list=create_matrix(test_documents,lexicon,label_2_index)
	print(len(train_data_list),len(train_label_list),len(train_label_list))
	print(len(test_data_list),len(test_label_list),len(test_index_list))
	print(train_data_list[1])
	print(train_label_list[1])
	# print(train_index_list[:20])
	print('-'*15)

	max_len=max(map(len,train_data_list+test_data_list))
	print('max_len:',max_len)  # 128
	min_len=min(map(len,train_data_list+test_data_list))
	print('min_len:',min_len)

	# 前面补0  padding
	print(train_data_list[0])
	train_data_array,train_label_list_padding=padding_sentences(train_data_list,train_label_list,max_len)
	print(train_data_array.shape)
	print(train_data_array[0])

	print(test_data_list[0])
	test_data_array,test_label_list_padding=padding_sentences(test_data_list,test_label_list,max_len)
	print(test_data_array.shape)
	print(test_data_array[0])

	# label 	
	# print(train_label_list_padding[0])
	train_label_array=np_utils.to_categorical(train_label_list_padding,len(label_2_index)+1).\
						reshape((len(train_label_list_padding),len(train_label_list_padding[0]),-1))
	print(train_label_array.shape)

	# label 	
	# print(test_label_list_padding[0])
	test_label_array=np_utils.to_categorical(test_label_list_padding,len(label_2_index)+1).\
						reshape((len(test_label_list_padding),len(test_label_list_padding[0]),-1))
	print(test_label_array.shape)

	

	# model
	model=Bilstm_CNN_Crf(max_len,len(lexicon),len(label_2_index)+1,embedding_weights)

	model.summary()

	print(model.input_shape)
	print(model.output_shape)


	plot_model(model, to_file='bilstm_cnn_crf_model.png',show_shapes=True,show_layer_names=True)

	train_nums=len(train_data_array)
	train_array,val_array=train_data_array[:int(train_nums*0.9)],train_data_array[int(train_nums*0.9):]
	train_label,val_label=train_label_array[:int(train_nums*0.9)],train_label_array[int(train_nums*0.9):]

	print(train_array.shape,train_label.shape)
	print(val_array.shape,val_label.shape)
	print(test_data_array.shape,test_label_array.shape)

	checkpointer=ModelCheckpoint(filepath='best_val_model.hdf5',verbose=1,\
				save_best_only=True,monitor='val_loss',mode='auto')

	# train model
	
	hist=model.fit(train_array,train_label,batch_size=256,epochs=20,verbose=1,\
				validation_data=(val_array,val_label),callbacks=[checkpointer])

	print(hist.history['val_loss'])
	best_model_epoch=np.argmin(hist.history['val_loss'])
	print('best_model_epoch:',best_model_epoch)

	# 可视化loss acc
	# plot_acc(hist)
	# print(hist.history)	
	

	model.load_weights('best_val_model.hdf5')

	test_y_pred=model.predict(test_data_array,batch_size=512,verbose=1)
	# print(test_y_pred)
	# 预测标签 [0,0,....,1,2,3,1]
	pred_label=np.argmax(test_y_pred,axis=2)
	print(pred_label[0])
	print(test_label_list_padding[0])

	K.clear_session()
	
	print(pred_label.shape,test_label_list_padding.shape)

	# 生成输出文档
	real_text_list,pred_text_list,real_label_list,pred_label_list=create_pred_text(\
		lexicon_reverse,test_data_array,pred_label,test_label_list_padding,test_index_list)

	'''
	for r_text,p_text,r_label,p_label in zip(real_text_list,pred_text_list,real_label_list,pred_label_list):
		print(r_text)
		print([index_2_label[r] for r in r_label])
		print('-'*10)
		print(p_text)
		print([index_2_label[p] for p in p_label])
		print('='*20)
	'''

	# 写文件
	write_2_file(real_text_list,pred_text_list)

	# score
	F=score.prf_score('real_text.txt','pred_text.txt',prf_file,text_seed,best_model_epoch)

	# F_list.append([text_seed,F])

	return F

def main():

	F_list=[]
	prf_file='prf_result_max_epoch_50_conll_law.txt'

	for text_seed in [1111*i for i in range(1,2)]:

		# step_1
		spath='biaozhu_1_100'
		t_path='corpus3'

		conll_path='conll2012_new'


		if not os.path.exists(t_path):
			os.mkdir(t_path)

		# text_seed=3333
		k=0.0

		F=process(spath,t_path,conll_path,text_seed,k,prf_file)

		F_list.append([text_seed,F])
		
	ave_f=sum([i for _,i in F_list])/len(F_list)
	print('ave_f:%.3f',ave_f)

if __name__ == '__main__':
	main()