# BiLstm_CNN_CRF_CWS
BiLstm+CNN+CRF 
[在线演示](http://118.25.42.251:7777/fenci?type=mine&text=%E5%8D%97%E4%BA%AC%E5%B8%82%E9%95%BF%E8%8E%85%E4%B8%B4%E6%8C%87%E5%AF%BC%EF%BC%8C%E5%A4%A7%E5%AE%B6%E7%83%AD%E7%83%88%E6%AC%A2%E8%BF%8E%E3%80%82%E5%85%AC%E4%BA%A4%E8%BD%A6%E4%B8%AD%E5%B0%86%E7%A6%81%E6%AD%A2%E5%90%83%E4%B8%9C%E8%A5%BF%EF%BC%81 "云服务器较烂 有时会崩")  

note: 实验基于

anaconda py3.5

tensorflow==1.0.1

keras==2.0.6

keras_contrib==2.0.8  pip install git+https://www.github.com/keras-team/keras-contrib.git

cuda==v8.0

gpu==GTX750Ti

# 简要介绍
![model](https://github.com/FanhuaandLuomu/BiLstm_CNN_CRF_CWS/blob/master/bilstm_cnn_crf_model.png)
1. 使用 bilstm+cnn+crf训练seq2seq模型
2. 预训练词向量 gensim
3. 段落有的太长，按简单标点切分为句子，maxlen控制在100+,不足maxlen前面补0
4. 测试也是按句子测试，最后还原成段落
5. _有机会写个blog，先准备过年~新年快乐！__

# step1: 法律文档+conll2012分词语料 训练word embedding
python embedding_model.py
# step2: 预处理+训练+测试
1. 随机抽80篇训练（10%用于验证集），20篇用于测试
   实验10次，平均**f-score=0.953**,详见prf_result_max_epoch_50_em.txt

2. 随机抽50篇训练（10%用于验证集），50篇用于测试
   实验10次，平均**f-score=0.933**,详见prf_result_max_epoch_50_law.txt
   
3. 用conll2012中6个领域的分词训练语料+法律文档训练语料（20篇）-> 法律80篇测试
   时间问题，只测一次：**f-score:0.943**
   
4. 用conll2012中6个领域的分词训练语料-> 法律100篇测试
   时间问题，只测一次：**f-score:0.757**
   
 # New 拖了好久，终于在毕业论文交（3.26）后写了一篇分词blog（虽然也没啥技术含量，写着玩..） 
    [基于BiLSTM-CNN-CRF的中文分词](https://www.jianshu.com/p/5fea8f42caa9 "简书链接") 
# simple_bilstm_model
程序写的太繁琐，简化了一下  只关心
pip install keras==2.0.6  深度学习分词算法的可以只看这个文件夹下的bilstm_cnn_crf.py程序
keras_contrib==2.0.8 pip install git+https://www.github.com/keras-team/keras-contrib.git  
pip install gensim  
如缺少其它模块，看报错自行安装  

	## note
	# 把你的语料放到corpus文件夹下  我的corpus中的语料压缩了，如使用可以解压
	# 1. python embedding_model.py  -> model_conll_law.m  生成词向量文件
	# 2. python bilstm_cnn_crf.py    // is_train==1
	# 会得到 train_model.hdf5  lexicon.pkl
	# 3. 可以在之前的基础上train_model.hdf5，继续训练
	# 4. 训练完成，测试  is_train==0
	# python bilstm_cnn_crf.py  按句测试或按文件测试

	# my_weights 中存放的是我的权值 

## 关于simple_bilstm_model程序的运行，写了个讲解，详见
[BiLSTM_CNN_CRF分词程序—运行讲解-简书](https://www.jianshu.com/p/373ce87e6f32 "简书链接")  
[BiLSTM_CNN_CRF分词程序—运行讲解-知乎](https://zhuanlan.zhihu.com/p/35710301 "知乎链接")

simple_bilstm_model 百度网盘下载：链接：https://pan.baidu.com/s/1b0WRe16aVVILYGEBmhB9lg 密码：9tiv  

不想下载全部项目的可以只下载网盘的内容。  


