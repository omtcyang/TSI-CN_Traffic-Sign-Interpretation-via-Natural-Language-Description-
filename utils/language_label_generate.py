import os
import os.path as osp
import json
import cv2
import numpy as np 
import matplotlib.pyplot as plt

from .symbol_affiliation import all_symbols


'''
生成 interpretation label
'''

def run(root, int_dir):
	language_label = []
	for item in os.listdir(root):  # 遍历 test_gt, training_gt 下的所有文件
		if 'json' in item:
			filepath = osp.join(root, item)  # 生成json文件的path

			with open(filepath,'r',encoding='utf8') as fp:
				json_data = json.load(fp)
				keys = list(json_data.keys())  # b0, b1, ...,bn
				for key in keys:
					if key == 'other': continue
					if not json_data[key]['affiliation']: continue  # 如果没有连接关系，则返回
					if json_data[key]['board']['ignore']==1: continue  # 如果指示牌需要忽略，则返回

					text, symbol, affiliation = \
						json_data[key]['text'],json_data[key]['symbol'],json_data[key]['affiliation']

					new_text, new_symbol = {}, {}
					for t in text:
						new_text[t['id']]=\
							{t['class']:
							[int((t['points'][0][0]+t['points'][2][0])/2),int((t['points'][0][1]+t['points'][2][1])/2)]}
					for s in symbol:
						new_symbol[s['id']]=\
							{all_symbols[s['class']]:
							[int((s['points'][0][0]+s['points'][2][0])/2),int((s['points'][0][1]+s['points'][2][1])/2)]}

					node = []
					summary = []
					for ak in list(affiliation.keys()):
						aff = affiliation[ak]

						head_id = aff['head']
						if 's' in head_id: node.append(new_symbol[head_id])
						elif 't' in head_id: node.append(new_text[head_id])

						for an in aff['node']:
							if 's' in an: node.append(new_symbol[an])
							elif 't' in an: node.append(new_text[an])

						summary.append(aff['string'])

					language = {'board_xy':json_data[key]['board']['points'],'content_node':node, 'summary':summary, 'img_name':item}
					# print(language)
					language_label.append(language)

	filename = root.split('_')[0] + '_language_label.txt'
	file_path = osp.join(int_dir, filename)
	# print(file_path,root,int_dir)
	with open(file_path,'at+',encoding='utf8') as dpth:
		for ll in language_label:
			dpth.write(str(ll)+'\n')
