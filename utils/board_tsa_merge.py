import os
import os.path as osp
import json
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from .symbol_affiliation import *
import Polygon as plg

'''
将 【原始标签文件夹：GT_board/, GT_tsa/】 里面的标签进行融合处理，存入 【目标标签文件夹： GT/】
'''


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG);


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


# 把逆时针标注的polygon调整至顺时针
def to_shunshizhen(coord):
	p_ori = plg.Polygon(coord)

	p1, p2, p3, p4 = [int(coord[0][0]),int(coord[0][1])], \
					 [int(coord[1][0]),int(coord[1][1])], \
					 [int(coord[2][0]),int(coord[2][1])], \
					 [int(coord[3][0]),int(coord[3][1])]

	suppose_shun_2_shun = [p1, [p2[0],p1[1]], [p2[0],p4[1]], [p1[0],p4[1]]]  # 假设轮廓点为顺时针，将点序用除p3外的其他点进行顺时针近似表示
	suppose_ni_2_shun = [p1, [p4[0],p1[1]], [p4[0],p2[1]], [p1[0],p2[1]]]  # 假设轮廓点为逆时针，将点序用除p3外的其他点进行顺时针近似表示

	p_s2s = plg.Polygon(suppose_shun_2_shun)
	p_n2s = plg.Polygon(suppose_ni_2_shun)

	inter_s2s = get_intersection(p_s2s, p_ori)
	inter_n2s = get_intersection(p_n2s, p_ori)

	# 通过判断确定是原轮廓点是顺时针还是逆时针
	if inter_s2s>=inter_n2s: return [p1,p2,p3,p4]
	else: return [p1,p4,p3,p2]


# 范围判断函数
# -1. 判断 text symbol affiliation 是否在 某个board里面
# -2. 判断 affiliation 涉及到哪些 text symbol
def is_include(obj_coord, range_coord):
	rcx_1, rcy_1, rcx_3, rcy_3 = range_coord[0][0], range_coord[0][1], range_coord[2][0], range_coord[2][1]

	if np.array(obj_coord).shape==(4,2):
		ocx_1, ocy_1, ocx_3, ocy_3 = obj_coord[0][0], obj_coord[0][1], obj_coord[2][0], obj_coord[2][1]
		x_mid, y_mid = (ocx_1+ocx_3)/2, (ocy_1+ocy_3)/2
		return x_mid>=rcx_1 and x_mid<=rcx_3 and y_mid>=rcy_1 and y_mid<=rcy_3

	if np.array(obj_coord).shape==(2,):
		ocx_one, ocy_one = obj_coord[0], obj_coord[1]
		return ocx_one>=rcx_1 and ocx_one<=rcx_3 and ocy_one>=rcy_1 and ocy_one<=rcy_3


def run(ori_path, dsp):
	orb, ortsa = ori_path[0], ori_path[1]

	for num, item in enumerate(os.listdir(ortsa)):  # 遍历 tsa 下的所有文件
		if 'json' in item:
			orb_path, ortsa_path = osp.join(orb, item), osp.join(ortsa, item)  # 生成json文件的path
			list_label_inboard, list_affiliation_inboard = [], []
			list_label_outboard = {
									'other':{
											'text':[],
											'symbol':[]
											}
									}

			# ========== -1. 指示牌标签处理 ==========
			if osp.exists(orb_path):
				with open(orb_path,'r',encoding='utf8') as bp:
					json_data = json.load(bp)
					label_infos = json_data['shapes']
					b_index=0
					for label_info in label_infos:
						board_class = label_info['label'][0]  # 解析label
						if board_class=='6': continue  # 忽略广告牌
						board_ignore = eval(label_info['label'][1])
						board_point = label_info['points']  # 解析label

						if len(board_point)==2:
							board_point = [[int(board_point[0][0]),int(board_point[0][1])], [int(board_point[1][0]),int(board_point[0][1])], 
										  [int(board_point[1][0]),int(board_point[1][1])], [int(board_point[0][0]),int(board_point[1][1])]]
						elif len(board_point)==4:
							board_point = to_shunshizhen(board_point)
						
						board_label = {'b'+str(b_index):{'board':{'class':board_class, 'ignore':board_ignore, 'points':board_point}, 
														'text':[],
														'symbol':[],
														'affiliation':[]
														}
													}

						list_label_inboard.append(board_label)
						list_affiliation_inboard.append({'b'+str(b_index):[]})
						b_index+=1

			# ========== -2. 标志、文本、关系标签处理 ==========
			if osp.exists(ortsa_path):
				with open(ortsa_path,'r',encoding='utf8') as tsap:
					json_data = json.load(tsap)
					label_infos = json_data['shapes']

					t_index, s_index = 0, 0
					for label_info in label_infos:
						sta_class = label_info['label']  # 解析label
						sta_point = label_info['points']  # 解析label
						
						# 判断当前标志属于 text symbol affiliation 中的哪一种
						switch = 0
						if sta_class == '###': switch = 1
						elif sta_class in all_symbols: switch = 2
						elif sta_class in affiliations: switch = 3
						elif sta_class[:2] in affiliations or sta_class[:3] in affiliations: switch = 4
						else: switch = 1

						# 按照标志的类型 将point点的形式进行处理
						if switch==1 or switch==2:
							if len(sta_point)==2:
								sta_point = [[int(sta_point[0][0]),int(sta_point[0][1])], [int(sta_point[1][0]),int(sta_point[0][1])], 
										    [int(sta_point[1][0]),int(sta_point[1][1])], [int(sta_point[0][0]),int(sta_point[1][1])]]
				
							elif len(sta_point)==4:
								sta_point = to_shunshizhen(sta_point)

						elif switch==3: sta_point = sta_point[0]
						elif switch==4: sta_point = sta_point[0]

						# 将 text symbol 以及 affiliation 分别存储到 list_label_inboard 和 list_affiliation_inboard 中
						is_inboard = False
						for ll, la in zip(list_label_inboard, list_affiliation_inboard):
							# board points
							ll_key = list(ll.keys())[0]
							board_pts = ll[ll_key]['board']['points']
							if is_include(sta_point, board_pts):
								if switch==1: ll[ll_key]['text'].append({'id':'t'+str(t_index),'class':sta_class, 'points':sta_point}); t_index+=1
								elif switch==2: ll[ll_key]['symbol'].append({'id':'s'+str(s_index),'class':sta_class, 'points':sta_point}); s_index+=1
								elif switch==3: la[ll_key].append({'aff':sta_class, 'points':sta_point})
								elif switch==4: la[ll_key].append({'aff_string':sta_class, 'points':sta_point})
								is_inboard=True
								break
						if not is_inboard: 
							if switch==1: list_label_outboard['other']['text'].append({'id':'t'+str(t_index),'class':sta_class, 'points':sta_point}); t_index+=1
							elif switch==2: list_label_outboard['other']['symbol'].append({'id':'s'+str(s_index),'class':sta_class, 'points':sta_point}); s_index+=1

			# list_label_inboard 和 list_affiliation_inboard 存储的标签进行融合
			for num, lla in enumerate(list_label_inboard):
				lla_key = list(lla.keys())[0]
				lla_text, lla_symbol, lla_affiliation = lla[lla_key]['text'], lla[lla_key]['symbol'], lla[lla_key]['affiliation']
				laf = list_affiliation_inboard[num][lla_key]

				# 把当前 board 里面的所有affiliation拿出来
				affiliation_proposals = {}
				for sub_laf in laf:
					sub_laf_keys, sub_laf_values = list(sub_laf.keys()), list(sub_laf.values())
					if sub_laf_keys[0]=='aff':
						ap_keys = list(affiliation_proposals.keys())
						if sub_laf_values[0] in ap_keys: continue
						affiliation_proposals[sub_laf_values[0]] = {'string':'','head':'','node':[]}

				# 把当前 board 里面的所有affiliation中的point信息换成对应text symbol的id
				for sub_laf in laf:
					sub_laf_keys, sub_laf_values = list(sub_laf.keys()), list(sub_laf.values())
					if sub_laf_keys[0]=='aff':
						node_pts = sub_laf_values[1]
						is_find=False
						for lt in lla_text:
							tid, tpts = lt['id'], lt['points']
							if is_include(node_pts, tpts):
								affiliation_proposals[sub_laf_values[0]]['node'].append(tid)
								is_find=True
								break
						if not is_find:
							for ls in lla_symbol:
								sid, spts = ls['id'], ls['points']
								if is_include(node_pts, spts): affiliation_proposals[sub_laf_values[0]]['node'].append(sid); break
					else:
						sub_laf_key = sub_laf_values[0].split(':')[0]
						affiliation_proposals[sub_laf_key]['string'] = sub_laf_values[0].split(':')[1]
						node_pts = sub_laf_values[1]

						is_find=False
						for lt in lla_text:
							tid, tpts = lt['id'], lt['points']
							if is_include(node_pts, tpts):
								affiliation_proposals[sub_laf_key]['head'] = tid
								is_find=True
								break
						if not is_find:
							for ls in lla_symbol:
								sid, spts = ls['id'], ls['points']
								if is_include(node_pts, spts): affiliation_proposals[sub_laf_key]['head'] = sid; break

				list_label_inboard[num][lla_key]['affiliation'] = affiliation_proposals

			# 存储标签
			generated_label = {}
			# in board
			for lb in list_label_inboard:
				key, value = list(lb.keys())[0], list(lb.values())[0]
				generated_label[key] = value
			# out board
			key_other, value_other = list(list_label_outboard.keys())[0], list(list_label_outboard.values())[0]
			generated_label[key_other] = value_other

			dst_path = osp.join(dsp, item)

			with open(dst_path,'w',encoding='utf8')as dpth:
				json.dump(generated_label,dpth,ensure_ascii=False,indent=2)


if __name__ == '__main__':

	dsp = 'GT_revision'
	if not osp.exists(dsp):  # 如果文件夹：GT_revision\ 不存在，则创建
		os.mkdir(dsp)
	orb, ortsa = 'GT_board_revision', 'GT_tsa_revision' 
	
	run([orb, ortsa], dsp)

