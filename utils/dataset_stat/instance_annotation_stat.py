import os
import os.path as osp

import numpy as np
from PIL import Image
import cv2
import math
import scipy.io as scio
import json
from copy import deepcopy as cdc
from symbol_affiliation import all_symbols


def is_english(char):
    if not ('a'<=char<='z' or 'A'<=char<='Z'):
        return False
    return True

def is_chinese(char):
    if not ('\u4e00'<=char<='\u9fff'):
        return False
    return True

def is_number(char):
    if char.isdigit():
        return True
    return False


def load_img(img_path):
    try:
        img = Image.open(img_path)
        img_shape = np.array(img).shape
        if len(img_shape) == 2:
            img = img.convert('RGB')
        if (len(img_shape) == 3) and (img_shape[2] == 4):
            img = img.convert('RGB')
        img = np.array(img)
    except Exception as e:
        raise ValueError('Failed loading img:', img_path)

    return img


def parse_NWPU_SA(data_dir, gt_dir):

    img_paths, filenames, imgs = [], [], []
    contours, classes, ignores = [], [], []
    affiliations = []

    for num, gt_file in enumerate(os.listdir(gt_dir)):

        # GT 路径
        img_file = str(gt_file.split('.')[0]) + '.jpg'
        img_path = osp.join(data_dir, img_file)
        # img = load_img(img_path)

        # 初始化 board, text, symbol 
        board_contours, board_classes, board_ignores = [], [], []
        text_contours, text_classes, text_ignores = [], [], []
        symbol_contours, symbol_classes, symbol_ignores = [], [], []

        gt_path = osp.join(gt_dir, gt_file) 
        with open(gt_path,'r',encoding='utf8') as gp:
            json_datas = json.load(gp)
            keys = list(json_datas.keys())  # b0, b1, ...,bn
            # print(keys)
            for key in keys:
                json_data = json_datas[key]
                # print(json_data)
                if key != 'other':
                    board_data, text_data, symbol_data, affiliation_data = \
                            json_data['board'], json_data['text'], json_data['symbol'], json_data['affiliation']

                    affiliations.append(affiliation_data) 
                    # 1. board label processing
                    board_class, board_ignore, board_points = \
                            board_data['class'], board_data['ignore'], board_data['points']
                    board_classes.append(board_class)
                    if board_ignore == 1:  # 如果指示牌需要被忽略，那里面的元素皆不参与训练
                        board_ignores.append(board_points) # board ignore: 忽略ignore=1的board
                        continue
                    board_contours.append(board_points)  
                 
                else: text_data, symbol_data = json_data['text'], json_data['symbol']
                    
                # 2. text label processing
                for td in text_data:
                    text_id, text_class, text_points = \
                        td['id'], td['class'], td['points']

                    text_classes.append(text_class)
                    if text_class == '###': 
                        text_ignores.append(text_points)  # text ignore: 忽略‘###’文本
                        continue

                    text_contours.append(text_points)

                # 3. symbol label processing
                for sd in symbol_data:
                    symbol_id, symbol_class, symbol_points = \
                            sd['id'], sd['class'], sd['points']
                    symbol_classes.append(symbol_class)
                    symbol_contours.append(symbol_points)
                    
            img_paths.append(img_path)
            filenames.append(img_path.split('/')[-1].split('.')[0])
            # imgs.append(img)

            contours.append([board_contours, text_contours, symbol_contours])
            classes.append([board_classes, text_classes, symbol_classes])
            ignores.append([board_ignores, text_ignores, symbol_ignores])
    

    return [img_paths, filenames, imgs, contours, classes, ignores, affiliations]


if __name__ == '__main__':
    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    # from config import r18_sa

    file_path = 'dataset_class_4_tongji.json'

    is_generate=1

    if is_generate:
        print('train_parse_NWPU_SA')
        _, _, _, train_contours, train_classes, train_ignore_masks, train_affiliations = \
                                parse_NWPU_SA('train/Image',
                                              'train/GT')

        instance_num, annotation_num = 0, 0
        affiliation_num = 0
        c = []

        for train_contour in train_contours:
            for tc in train_contour:
                instance_num+=len(tc)
        print('instance_num',instance_num)

        for train_ignore_mask in train_ignore_masks:
            for tim in train_ignore_mask:
                instance_num+=len(tim)
        print('instance_num',instance_num)


        for train_class in train_classes:
            for tc in train_class:
                annotation_num+=len(tc)
        print('annotation_num',annotation_num)

        for train_affiliation in train_affiliations:
            affiliation_num+=len(list(train_affiliation.keys()))
        print('affiliation_num',affiliation_num)

        text_class = train_classes[1]
        for tcs in train_classes:
            text_class = tcs[1]
            for tc in text_class:
                for t in tc:
                    for tt in t:
                        c.append(tt)

        for train_affiliation in train_affiliations:
            keys = list(train_affiliation.keys())
            for key in keys:
                aff = train_affiliation[key]['string']
                for af in aff:
                    c.append(af)




        print('test_parse_NWPU_SA')
        _, _, _, test_contours, test_classes, test_ignore_masks, test_affiliations = \
                                parse_NWPU_SA('test/Image',
                                              'test/GT')


        for test_contour in test_contours:
            for tc in test_contour:
                instance_num+=len(tc)
        print('instance_num',instance_num)

        for test_ignore_mask in test_ignore_masks:
            for tim in test_ignore_mask:
                instance_num+=len(tim)
        print('instance_num',instance_num)


        for test_class in test_classes:
            for tc in test_class:
                annotation_num+=len(tc)
        print('annotation_num',annotation_num)



        for test_affiliation in test_affiliations:
            affiliation_num+=len(list(test_affiliation.keys()))
        print('affiliation_num',affiliation_num)




        text_class = test_classes[1]
        for tcs in test_classes:
            text_class = tcs[1]
            for tc in text_class:
                for t in tc:
                    for tt in t:
                        c.append(tt)

        for test_affiliation in test_affiliations:
            keys = list(test_affiliation.keys())
            for key in keys:
                aff = test_affiliation[key]['string']
                for af in aff:
                    c.append(af)
        print('c',len(c))


    CN, EN, NUM, O = [], [], [], []
    for cc in c:
        if is_english(cc): EN.append(cc)
        elif is_chinese(cc): CN.append(cc)
        elif is_number(cc): NUM.append(cc)
        else:O.append(cc)
    print('CN, EN, NUM, O', len(CN), len(EN), len(NUM), len(O))


