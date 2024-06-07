import os
import os.path as osp
import json
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import Polygon as plg

'''
把json文件里的二进制图像信息删掉，因为用labelme标注的过程中会把图像信息保存进去，
可能导致标签名称和图像信息名称对应不起来，从而标签混乱
'''

def run(ori_path, dst_path):
	for op, dp in zip(ori_path, dst_path):

		if not osp.exists(dp):
			os.mkdir(dp)

		for item in os.listdir(op):  
			if 'json' not in item: continue

			filename = item.split('.')[0]
			filename_jpg = filename + '.jpg'

			json_data = []
			with open(osp.join(op, item),'r',encoding='utf8') as opf:
				json_data = json.load(opf)
				image_path = json_data['imagePath']
				json_data['imagePath'] = filename_jpg
				json_data['imageData'] = None

			with open(osp.join(dp, item),'w',encoding='utf8') as dpf:
				# print(item,filename_jpg,json_data['imagePath'],json_data['imageData'])
				json.dump(json_data,dpf,ensure_ascii=False,indent=2)


if __name__ == '__main__':

	ori_path = ['GT_board','GT_tsa']
	dst_path = ['GT_board_revision','GT_tsa_revision']
	run(ori_path, dst_path)


