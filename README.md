# BiLstm_CNN_CRF_CWS
BiLstm+CNN+CRF  法律文档（合同类案件）领域分词（100篇标注样本）
note: 实验基于
anaconda py3.5

tensorflow==1.0.1

keras==2.0.6

keras_contrib==2.0.8

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
   实验10次，平均**f-score=0.9--**,详见prf_result_max_epoch_50_law.txt
3. 用conll2012中6个领域的分词训练语料+法律文档训练语料（20篇）-> 法律80篇测试
   时间问题，只测一次：**f-score:0.9--**
4. 用conll2012中6个领域的分词训练语料-> 法律100篇测试
   时间问题，只测一次：**f-score:0.9--**

