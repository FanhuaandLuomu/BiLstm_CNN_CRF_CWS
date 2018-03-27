#coding:utf-8
import requests
import itchat
import jieba
jieba.initialize()

from fenci_server import word_seg
from lss_fenci_api import samme_cws


@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
	text=msg.text.strip()
	
	if text.startswith('jieba'):
		cut_type='jieba'
		text=text[5:]
	elif text.startswith('samme'):
		cut_type='samme'
		text=text[5:]
	else:
		cut_type='mine'
	
	# reply=requests.get('http://127.0.0.1:7777/cut?type=%s&text=%s' %(cut_type,text)).content

	# reply=word_seg(text)[0]

	#print reply.decode('utf-8').encode('GB18030')
	# return reply.decode('utf-8')

	if cut_type=='jieba':
		print('jieba')
		return ' '.join(jieba.cut(text.strip())).strip()+'\n——[By结巴分词]'
	if cut_type=='samme':
		print('samme')
		return samme_cws(text.strip())[0].strip()+'\n——[By samme分词]'
	else:
		print('mine')
		return word_seg(text.strip())[0].strip()+'\n——[By我的分词]'

	return reply

itchat.auto_login()
itchat.run()
