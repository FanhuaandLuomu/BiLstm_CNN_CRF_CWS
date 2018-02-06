#coding:utf-8
# py3.5+tensorflow-1.0.1+keras-2.0.6
# seq2seq bilstm+cnn+crf
import gc
import numpy as np
# np.random.seed(1111)

import keras
from keras.layers import *
from keras.models import *
from keras_contrib.layers import CRF

from keras import backend as K

from keras.utils import plot_model
from keras.utils import np_utils

# input:
# maxlen  char_value_dict_len  class_label_count
def Bilstm_CNN_Crf(maxlen,char_value_dict_len,class_label_count):
	word_input=Input(shape=(maxlen,),dtype='int32',name='word_input')
	word_emb=Embedding(char_value_dict_len+2,output_dim=64,\
				input_length=maxlen,name='word_emb')(word_input)
	
	# bilstm
	bilstm=Bidirectional(LSTM(32,return_sequences=True))(word_emb)
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

maxlen,char_value_dict_len,class_label_count=100,1000,4
model=Bilstm_CNN_Crf(maxlen,char_value_dict_len,class_label_count)
model.summary()

print(model.input_shape)
print(model.output_shape)

plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=True)

# train
x_train=np.random.randint(0,1000,(500,100))
y_train=np.random.randint(0,4,(500,100))
y_train=np_utils.to_categorical(y_train,4)

model.fit(x_train,y_train,batch_size=16,epochs=10,verbose=1)

# test
test_data=np.random.randint(0,1000,(10,100))
test_y_pred=model.predict(test_data)
print(test_y_pred)
print(np.argmax(test_y_pred,axis=2))

K.clear_session()