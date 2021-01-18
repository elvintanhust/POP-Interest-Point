import argparse
import warnings
import os
from evaluation_main import EvaluationMain

warnings.filterwarnings("ignore")


def eval(args):
    write_image_path = args.matching_image_save_path
    need_write_img = (write_image_path is not None)

    # all methods
    mathod_info_list = []
    # POP
    resp_thre = 0.5
    mathod_info_now = {}
    mathod_info_now['method_name'] = 'POP'
    mathod_info_now['resp_thre'] = resp_thre
    mathod_info_now['para_dict'] = {
        'our_model_name': args.POP_model_path,
        'eval_match': True,
        'write_filename': 'POP_net'}
    mathod_info_list.append(mathod_info_now)

    # 对比算法
    if args.eval_comparison_methods:
        # SuperPoint
        mathod_info_now = {}
        mathod_info_now['method_name'] = 'superpoint'
        # use the threshold recommended by the author
        mathod_info_now['resp_thre'] = 0.015
        mathod_info_now['para_dict'] = {'eval_match': True}
        mathod_info_list.append(mathod_info_now)
        # SIFT
        mathod_info_now = {}
        mathod_info_now['method_name'] = 'SIFT'
        mathod_info_now['resp_thre'] = None
        mathod_info_now['para_dict'] = {'eval_match': True,
                                        'write_filename': 'SIFT'}
        mathod_info_list.append(mathod_info_now)
        # ORB
        mathod_info_now = {}
        mathod_info_now['method_name'] = 'ORB'
        mathod_info_now['resp_thre'] = None
        mathod_info_now['para_dict'] = {'eval_match': True,
                                        'write_filename': 'ORB'}
        mathod_info_list.append(mathod_info_now)

    # the information of dataset
    dataset_info_list = []
    dataset_info_now = {}
    dataset_info_now['data_dir'] = args.HPatches_path
    dataset_info_now['name'] = os.path.basename(args.HPatches_path)
    dataset_info_list.append(dataset_info_now)

    # 图像尺寸和最大特征点数目
    image_info_list = []
    image_info_now = {'image_row': 480, 'image_col': 640, 'max_point_num': 1000}
    image_info_list.append(image_info_now)

    # soft_dist设定
    soft_dist_match_list = [1, 3]

    for mathod_id, mathod_info in enumerate(mathod_info_list):
        print('\n--------start the evaluation of %s--------' % (mathod_info['method_name']))
        for dataset_id, dataset_info in enumerate(dataset_info_list):
            data_dir = dataset_info['data_dir']
            for image_info in image_info_list:
                for soft_dist in soft_dist_match_list:
                    image_row, image_col = image_info['image_row'], image_info['image_col']
                    max_point_num = image_info['max_point_num']

                    nms_rad = 4
                    out_dist = 7
                    method_fullname = mathod_info['method_name']
                    if 'write_filename' in mathod_info['para_dict'].keys():
                        method_fullname = mathod_info['para_dict']['write_filename']
                    if 'link_mark' in mathod_info['para_dict'].keys() and \
                            mathod_info['para_dict']['link_mark']:
                        method_fullname = 'ours_' + method_fullname

                    evaluator = EvaluationMain(args.device)

                    if need_write_img:
                        match_image_path = os.path.join(
                            write_image_path, method_fullname, dataset_info['name'])
                    else:
                        match_image_path = None
                    evaluator.set_dataset(data_dir, match_image_path=match_image_path)
                    evaluator.set_hyper(nms_rad, soft_dist, out_dist, max_point_num,
                                        mathod_info['resp_thre'], image_row, image_col)

                    result_mean, result_mat, result_item, time_mean = evaluator.main(
                        mathod_info['method_name'],
                        mathod_info['para_dict'])

                    result_filename = os.path.join(args.statistics_save_path,
                                                   method_fullname + '.txt')
                    result_info_str = (
                        'dataset: %s, image_row: %d, image_col: %d, max_point_num: %d, '
                        'epsilon: %d, resp_thre:%.2f, ' %
                        (dataset_info['name'], image_info['image_row'],
                         image_info['image_col'], image_info['max_point_num'], soft_dist,
                         resp_thre))
                    result_info_str += ('nms_rad: %d, out_dist: %d' % (nms_rad, out_dist))
                    result_str = \
                        result_item[0] + ':%.5f, ' % result_mean[0] + \
                        result_item[1] + ':%.5f, ' % result_mean[1] + \
                        result_item[2] + ':%.5f, ' % result_mean[2] + \
                        result_item[3] + ':%.5f, ' % result_mean[3] + \
                        result_item[4] + ':%.5f, ' % result_mean[4] + \
                        'time:%.5f' % time_mean

                    with open(result_filename, 'a') as file_to_write:
                        file_to_write.write('\n-------------------\n\n')
                        file_to_write.write(result_info_str)
                        file_to_write.write('\n\n')
                        file_to_write.write(result_str)
                        file_to_write.write('\n')


def main():
    parser = argparse.ArgumentParser(description="The evaluation of POP and compared methods")

    # root_dir = 'hpatches-sequences-release'
    # root_dir = '/home/ubuntu/yanpei/data/data_yanpei/hpatches-i'
    parser.add_argument('--HPatches-path', type=str,
                        default='hpatches-sequences-release',
                        help='the path of hpatches sequences dataset')
    parser.add_argument('--POP-model-path', type=str,
                        default='save_POP_model/POP_net_pretrained.pth',
                        help='set the path of the POP network')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='the device used to perform the model, ' \
                             'which is represented with the pytorch format')
    parser.add_argument('--eval-comparison-methods', action='store_true', default=True,
                        help='whether evaluate the performance of the comparison methods')
    parser.add_argument('--statistics-save-path', type=str,
                        default='statistics_results',
                        help='the path to save the statistics results')
    parser.add_argument('--matching-image-save-path', type=str,
                        default=None,
                        help='the path to save the image matching results, ' \
                             'which requires some spaces to store the results of every image pair.')

    args = parser.parse_args()

    # create the storage directory if needed
    if not os.path.exists(args.statistics_save_path):
        os.mkdir(args.statistics_save_path)
    if args.matching_image_save_path is not None:
        if os.path.exists(args.matching_image_save_path):
            os.mkdir(args.matching_image_save_path)

    eval(args)


if __name__ == '__main__':
    main()
