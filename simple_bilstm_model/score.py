#coding:utf_8
import sys
import codecs
# """
#   通过与黄金标准文件对比分析中文分词效果.

#   使用方法：
#           python crf_tag_score.py test_gold.utf8 your_tagger_output.utf8

#   分析结果示例如下:
#     标准词数：104372 个，正确词数：96211 个，错误词数：6037 个
#     标准行数：1944，正确行数：589 ，错误行数：1355
#     Recall: 92.1808531024%
#     Precision: 94.0957280338%
#     F MEASURE: 93.1284483593%


#   参考：中文分词器分词效果的评测方法
#   http://ju.outofmemory.cn/entry/46140

# """
'''
通过与黄金标准文件对比分析中文分词效果.
分析结果如下：
一次迭代：
result:
标准词数：26940个，正确词数：25341个，错误词数：1739个
标准行数：929，正确行数：407，错误行数：522
Recall: 0.940646
Precision: 0.935783
F MEASURE: 0.938208
ERR RATE: 0.064551

三十次迭代:
result:
标准词数：26940个，正确词数：25719个，错误词数：1299个
标准行数：929，正确行数：493，错误行数：436
Recall: 0.954677
Precision: 0.951921
F MEASURE: 0.953297
ERR RATE: 0.048218


CRF:
result:
标准词数：26940个，正确词数：25544个，错误词数：1354个
标准行数：929，正确行数：480，错误行数：449
Recall: 0.948181
Precision: 0.949662
F MEASURE: 0.948921
ERR RATE: 0.050260

'''
def read_line(f):
    '''
        读取一行，并清洗空格和换行 
    '''
    line = f.readline()
    line = line.strip('\n').strip('\r').strip(' ')
    while (line.find('  ') >= 0):
        line = line.replace('  ', ' ')
    return line


def prf_score(real_text_file,pred_text_file,prf_file,seed,best_epoch):
    file_gold = codecs.open(real_text_file, 'r', 'utf8')
    # file_gold = codecs.open(r'../corpus/msr_test_gold.utf8', 'r', 'utf8')
    # file_tag = codecs.open(r'pred_standard.txt', 'r', 'utf8')
    file_tag = codecs.open(pred_text_file, 'r', 'utf8')

    line1 = read_line(file_gold)
    N_count = 0   #将正类分为正或者将正类分为负
    e_count = 0   #将负类分为正
    c_count = 0   #正类分为正
    e_line_count = 0
    c_line_count = 0
                                                                                                                                                                                                                           
    while line1:
        line2 = read_line(file_tag)

        list1 = line1.split(' ')
        list2 = line2.split(' ')

        count1 = len(list1)   # 标准分词数
        N_count += count1
        if line1 == line2:
            c_line_count += 1#分对的行数
            c_count += count1#分对的词数
        else:
            e_line_count += 1
            count2 = len(list2)

            arr1 = []
            arr2 = []

            pos = 0
            for w in list1:
                arr1.append(tuple([pos, pos + len(w)]))#list1中各个单词的起始位置
                pos += len(w)

            pos = 0
            for w in list2:
                arr2.append(tuple([pos, pos + len(w)]))#list2中各个单词的起始位置
                pos += len(w)

            for tp in arr2:
                if tp in arr1:
                    c_count += 1
                else:
                    e_count += 1

        line1 = read_line(file_gold)

    R = float(c_count) / N_count
    P = float(c_count) / (c_count + e_count)
    F = 2. * P * R / (P + R)
    ER = 1. * e_count / N_count

    #print '  标准词数：{} 个，正确词数：{} 个，错误词数：{} 个'.format(N_count, c_count, e_count).decode('utf8')
    # print '  标准行数：{}，正确行数：{} ，错误行数：{}'.format(c_line_count+e_line_count, c_line_count, e_line_count).decode('utf8')
    # print '  Recall: {}%'.format(R)
    # print '  Precision: {}%'.format(P)
    # print '  F MEASURE: {}%'.format(F)
    # print '  ERR RATE: {}%'.format(ER)
    print("result:")
    print('标准词数：%d个，正确词数：%d个，错误词数：%d个' %(N_count, c_count, e_count))
    print('标准行数：%d，正确行数：%d，错误行数：%d'%(c_line_count+e_line_count, c_line_count, e_line_count))
    print('Recall: %f'%(R))
    print('Precision: %f'%(P))
    print('F MEASURE: %f'%(F))
    print('ERR RATE: %f'%(ER))

    #print P,R,F

    f=codecs.open(prf_file,'a','utf-8')
    f.write('result-(seed:%s , best_epoch:%s):\n' %(seed,best_epoch))
    f.write('标准词数：%d个，正确词数：%d个，错误词数：%d个\n' %(N_count, c_count, e_count))
    f.write('标准行数：%d，正确行数：%d，错误行数：%d\n'%(c_line_count+e_line_count, c_line_count, e_line_count))
    f.write('Recall: %f\n'%(R))
    f.write('Precision: %f\n'%(P))
    f.write('F MEASURE: %f\n'%(F))
    f.write('ERR RATE: %f\n'%(ER))
    f.write('====================================\n')

    return F

def main():
    # prf_score('real_text.txt','corpus/test_p0.19999999999999996_11110.utf8','prf_tmp.txt',1111)

    pred_file='test_documents/conll2012_test_mine/wb_conll_testing.utf8'
    gold_file='conll2012_test_gold/wb_conll_testing.utf8'

    F=prf_score(pred_file,gold_file,'prf_tmp.txt',1111,10)

if __name__ == '__main__':
    main()
    