#coding:utf-8
import os
import codecs
import random

def create_file(spath,filenames,save_name):
	contents=[]
	for fname in filenames:
		f=codecs.open(spath+os.sep+fname,'r','utf-8')
		lines=f.readlines()
		f.close()
		contents+=lines

	f=codecs.open(save_name,'w','utf-8')
	f.write(''.join(contents))
	f.close()

def main():
	spath='biaozhu_1_100'
	filenames=os.listdir(spath)

	seed=2222
	random.seed(seed)
	random.shuffle(filenames)

	train_files=filenames[:80]
	test_files=filenames[20:]

	t_path='corpus'

	create_file(spath,train_files,t_path+os.sep+'train_p80_%d.utf8' %(seed))
	create_file(spath,test_files,t_path+os.sep+'test_p20_%d.utf8' %(seed))

# if __name__ == '__main__':
# 	main()