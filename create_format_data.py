# coding:utf-8
# 语料预处理 生成训练-测试数据
import os
import codecs

def process(s_file_list,t_file):
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

def main():
	seed=2222
	raw_train_file='corpus/train_p80_%d.utf8' %(seed)
	raw_test_file='corpus/test_p20_%d.utf8' %(seed)

	process([raw_train_file],'train_%d.data' %seed)
	process([raw_test_file],'test_%d.data' %seed)

# if __name__ == '__main__':
# 	main()