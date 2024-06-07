import os
import os.path as osp
import shutil

from utils.test_train_index import test_index, train_index
from utils.del_image_info_from_label import run as pre_run_1
from utils.board_tsa_merge import run as pre_run_2
from utils.language_label_generate import run as pre_run_3
from utils.classes_label_generate import run as pre_run_5


if __name__ == '__main__':

	# step 1. delete image infomation from ori label
	print('step 1 --> delete image infomation from ori label')
	ori_label_path = ['GT_board','GT_tsa']
	dst_label_path = ['GT_board_revision','GT_tsa_revision']
	pre_run_1(ori_label_path, dst_label_path)

	# step 2. merge board and tsa labels
	print('step 2 --> merge board and tsa labels')
	dsp_label_merge = 'GT_revision'
	if osp.exists(dsp_label_merge): shutil.rmtree(dsp_label_merge); os.makedirs(dsp_label_merge)
	else: os.mkdir(dsp_label_merge)
	pre_run_2(dst_label_path, dsp_label_merge)
	for dlp in dst_label_path:
		shutil.rmtree(dlp)

	# step 3. split data to generate training set and test set
	print('step 3 --> split data to generate training set and test set')
	img_label_path = ['trianing_set', 'test_set']
	img_path, label_path = 'Image', dsp_label_merge
	data_index = [train_index, test_index]
	for ilp, di in zip(img_label_path, data_index):
		img_root, gt_root = osp.join(ilp, img_path), osp.join(ilp, 'GT')
		if osp.exists(img_root): shutil.rmtree(img_root); os.makedirs(img_root)
		else: os.makedirs(img_root)
		if osp.exists(gt_root): shutil.rmtree(gt_root); os.makedirs(gt_root)
		else: os.makedirs(gt_root)

		for ind in di:
			img_name, gt_name = 'IMG_' + ind + '.jpg', 'IMG_' + ind + '.json'
			ori_img, ori_gt = osp.join(img_path, img_name), osp.join(label_path, gt_name)
			dst_img, dst_gt = osp.join(img_root, img_name), osp.join(gt_root, gt_name)
			shutil.copy2(ori_img, dst_img)
			shutil.copy2(ori_gt, dst_gt)
	shutil.rmtree(dsp_label_merge)

	# step 4. generate language label for interpretation
	print('step 4 --> generate language label for interpretation')
	int_dir = 'GT_language/'
	if osp.exists(int_dir): shutil.rmtree(int_dir); os.makedirs(int_dir)
	else: os.mkdir(int_dir)
	for ilp in img_label_path:
		gt_root = osp.join(ilp, 'GT')
		pre_run_3(gt_root, int_dir)


	# step 5. generate classes label
	print('step 5 --> generate classes label')
	file_path = 'dataset_class.json'
	if osp.exists(file_path): shutil.rmtree(file_path)
	input_list = []
	for ilp in img_label_path:
		img_root, gt_root = osp.join(ilp, img_path), osp.join(ilp, 'GT')
		split = ilp.split('_')[0]
		input_list.append([img_root, gt_root, split])
	input_list.append(file_path)
	pre_run_5(input_list)




    





