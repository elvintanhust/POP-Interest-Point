import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import os
import math
from point_select_utils import remove_out_point


# compute the performance metrics on a sequence of images

class EvaluationSeq():
    def __init__(self, point_list: list, desc_list: list, H: np.ndarray,
                 soft_dist: int, out_dist: int,
                 image_row: int, image_col: int, image_shape_ori: np.ndarray,
                 need_HAMMING: bool, match_image_path: str = None):
        self.each_num = len(point_list)
        self.image_row, self.image_col = image_row, image_col
        self.xy_range = np.array([out_dist, self.image_col - out_dist,
                                  out_dist, self.image_row - out_dist])
        # soft_dist是特征点匹配时的容许距离
        self.soft_dist_match = soft_dist
        # soft_dist_homo是评价单应变换正确性时的容许距离
        # self.soft_dist_homo = 3
        self.soft_dist_homo = soft_dist
        self.image_shape_ori = image_shape_ori

        self.H = H

        self.point_list = point_list
        self.desc_list = desc_list

        self.need_HAMMING = need_HAMMING

        self.match_image_path = match_image_path

        # 用于随机采样一致性过程的重复次数
        self.try_num = 5
        # 统计局部点数目时，划分网格的尺寸
        self.grid_size = 120
        self.large_point_num_ratio = 0.5

    def map_point_ori_shape(self, point_list: list, image_shape_ori: torch.Tensor):
        each_num = len(point_list)
        point_list_mapped = copy.deepcopy(point_list)
        if image_shape_ori.shape[0] < 1:
            return point_list_mapped

        for image_id in range(each_num):
            shape_ori_now = image_shape_ori[image_id]
            point_ratio = np.array([[shape_ori_now[1] / self.image_col,
                                     shape_ori_now[0] / self.image_row]])
            point_list_mapped[image_id][:, :2] *= point_ratio
        return point_list_mapped

    def grid_num_large_ratio(self, image_row, image_col, point):
        # 函数功能：对图像进行网格划分，统计网格中点数大于阈值的数目

        # 得到网格划分，为每个网格分配id
        grad_r_num = int(round(image_row / self.grid_size))
        grad_row = int(math.floor(image_row / grad_r_num))
        grad_c_num = int(round(image_col / self.grid_size))
        grad_col = int(math.floor(image_col / grad_c_num))
        grad_id_mat = -np.ones((image_row, image_col), dtype='int')
        row_id_mat = np.arange(grad_r_num).reshape((grad_r_num, 1))
        row_id_mat = np.repeat(row_id_mat, grad_row, axis=0)
        col_id_mat = np.arange(grad_c_num).reshape((1, grad_c_num))
        col_id_mat = np.repeat(col_id_mat, grad_col, axis=1)
        grad_id_mat_exist = row_id_mat * grad_c_num + col_id_mat
        grad_id_mat[:grad_id_mat_exist.shape[0], :grad_id_mat_exist.shape[1]] = grad_id_mat_exist
        # 得到每个点对应的网格id
        point_int = np.floor(point).astype('int')
        point_belong_id = grad_id_mat[point_int[:, 1], point_int[:, 0]]
        # 统计每个网格内的点数目
        grad_num = grad_r_num * grad_c_num
        grad_id_bins = np.arange(grad_num + 1)
        grad_each_num, _ = np.histogram(point_belong_id, bins=grad_id_bins)
        # 得到高于数目阈值的网格数量
        point_num = point.shape[0]
        point_num_thre = point_num / grad_num * self.large_point_num_ratio
        num_large_ratio = np.sum(grad_each_num >= point_num_thre) / grad_num

        return num_large_ratio

    def get_pair_repeat(self, point1_img, point2_img, shape_ori_1, shape_ori_2, H_1_2):
        point1 = point1_img[:, :2].copy()
        point2 = point2_img[:, :2].copy()
        point1_num = point1.shape[0]
        point2_num = point2.shape[0]
        point_num = (point1_num + point2_num) / 2
        if point1_num < 1 or point2_num < 1:
            return 0, point_num

        point_ratio1 = np.array([[shape_ori_1[1] / self.image_col,
                                  shape_ori_1[0] / self.image_row]])
        point_ratio2 = np.array([[shape_ori_2[1] / self.image_col,
                                  shape_ori_2[0] / self.image_row]])
        # 1 to 2
        point_here = point1 * point_ratio1
        point_here = point_here.astype('float32')[np.newaxis, :]
        point1_warped = cv2.perspectiveTransform(point_here, H_1_2)
        point1_warped = point1_warped.squeeze(0)
        point1_warped /= point_ratio2
        point1_warped = remove_out_point(point1_warped, self.xy_range)
        # 2 to 1
        point_here = point2 * point_ratio2
        point_here = point_here.astype('float32')[np.newaxis, :]
        point2_warped = cv2.perspectiveTransform(point_here, np.linalg.inv(H_1_2))
        point2_warped = point2_warped.squeeze(0)
        point2_warped /= point_ratio1
        point2_warped = remove_out_point(point2_warped, self.xy_range)

        # Compute the repeatability
        N1 = point1_warped.shape[0]
        N2 = point2_warped.shape[0]
        point1 = np.expand_dims(point1, 1)
        point2_warped = np.expand_dims(point2_warped, 0)
        point2 = np.expand_dims(point2, 1)
        point1_warped = np.expand_dims(point1_warped, 0)
        # shapes are broadcasted to N1 x N2 x 2:
        dist1 = np.linalg.norm(point1 - point2_warped,
                               ord=None, axis=2)
        dist2 = np.linalg.norm(point2 - point1_warped,
                               ord=None, axis=2)
        count1 = 0
        count2 = 0
        repeatability = 0
        if N2 != 0:
            min1 = np.min(dist1, axis=1)
            count1 = np.sum(min1 <= self.soft_dist_match)
        if N1 != 0:
            min2 = np.min(dist2, axis=1)
            count2 = np.sum(min2 <= self.soft_dist_match)
        if N1 + N2 > 0:
            repeatability = (count1 + count2) / (N1 + N2)

        return repeatability, point_num

    def get_repeat(self):
        # 每对图像之间，计算重复率
        assert (self.each_num >= 2)
        pair_num = 0
        point_sum = 0
        repeat_sum = 0
        for id_1 in range(self.each_num):
            for id_2 in range(id_1 + 1, self.each_num):
                H_1_ref = np.linalg.inv(self.H[id_1])
                H_ref_2 = self.H[id_2]
                H_1_2 = np.dot(H_ref_2, H_1_ref)
                shape_ori_1 = self.image_shape_ori[id_1]
                shape_ori_2 = self.image_shape_ori[id_2]
                repeat_now, point_num_now = self.get_pair_repeat(self.point_list[id_1],
                                                                 self.point_list[id_2],
                                                                 shape_ori_1, shape_ori_2, H_1_2)
                repeat_sum += repeat_now
                point_sum += point_num_now
                pair_num += 1
        return repeat_sum / pair_num, point_sum / pair_num

    def get_pair_match_score(self, point1_img, point2_img, shape_ori_1, shape_ori_2,
                             feature1, feature2, real_H):
        point1 = point1_img[:, :2].copy()
        point2 = point2_img[:, :2].copy()
        point1_num = point1.shape[0]
        point2_num = point2.shape[0]
        match_score = 0

        if point1_num < 1 or point2_num < 1:
            return match_score, 0

        point_ratio1 = np.array([[shape_ori_1[1] / self.image_col,
                                  shape_ori_1[0] / self.image_row]])
        point_ratio2 = np.array([[shape_ori_2[1] / self.image_col,
                                  shape_ori_2[0] / self.image_row]])
        point1 = point1 * point_ratio1
        point2 = point2 * point_ratio2

        # 计算match score时不使用crossCheck
        if self.need_HAMMING:
            feature1 = feature1.astype(np.uint8)
            feature2 = feature2.astype(np.uint8)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # 两个方向的最近邻匹配结果均需计算，而后取均值
        score1, true_num1 = self.calcu_score_from_desc_pair(feature1, feature2, bf, point1, point2,
                                                            real_H, point_ratio2)
        score2, true_num2 = self.calcu_score_from_desc_pair(feature2, feature1, bf, point2, point1,
                                                            np.linalg.inv(real_H), point_ratio1)

        true_num = (true_num1 + true_num2) / 2
        if score1 < 0 or score2 < 0:
            return -1, true_num
        match_score = (score1 + score2) / 2

        return match_score, true_num

    def calcu_score_from_desc_pair(self, feature1, feature2, bf_obj, point1, point2,
                                   real_H, point_ratio2):
        # 若不存在公共区域的点，则返回-1，表示忽略该结果
        # 得到匹配点对
        matches = bf_obj.match(feature1, feature2)
        matches_idx = np.array([m.queryIdx for m in matches])
        point1_matched = point1[matches_idx, :]
        matches_idx = np.array([m.trainIdx for m in matches])
        point2_matched = point2[matches_idx, :]
        # 计算图像1中的点在图像2中的点的真实值
        point1_here = point1_matched.astype('float32')[np.newaxis, :]
        point2_true = cv2.perspectiveTransform(point1_here, real_H)
        point2_true = point2_true.squeeze(0)
        # 缩放到用于计算特征点的图像尺寸下
        point2_matched = point2_matched / point_ratio2
        point2_true = point2_true / point_ratio2
        # 忽略有效范围以外的点
        point2_here = np.c_[point2_true, point2_matched]
        point2_inner = remove_out_point(point2_here, self.xy_range)
        point2_true = point2_inner[:, :2]
        point2_matched = point2_inner[:, 2:]
        # 统计与真实值的距离小于阈值的点数目
        dist_vec = np.linalg.norm(point2_matched - point2_true, axis=1)
        match_num = np.sum(dist_vec < self.soft_dist_match)
        point2_true_inner = remove_out_point(point2_true, self.xy_range)
        exist_num = point2_true_inner.shape[0]
        if exist_num == 0:
            return -1, 0
        match_score = match_num / exist_num
        return match_score, match_num

    def get_pair_homography(self, point1_img, point2_img, shape_ori_1, shape_ori_2,
                            feature1, feature2, real_H,
                            write_match_mark=False,
                            image1: np.ndarray = None, image2: np.ndarray = None):
        point1 = point1_img[:, :2].copy()
        point2 = point2_img[:, :2].copy()

        point1_num = point1.shape[0]
        point2_num = point2.shape[0]

        draw_radius = 3
        draw_margin = 10
        white_image = image1[:, :draw_margin, :].copy()
        white_image[:] = 255

        if write_match_mark:
            show_image1 = image1.copy()
            show_image2 = image2.copy()
            point1_int = np.round(point1).astype('int')
            point2_int = np.round(point2).astype('int')
            for p_now in point1_int:
                show_image1 = cv2.circle(show_image1, (p_now[0], p_now[1]), radius=draw_radius,
                                         color=(0, 0, 255), thickness=-1)
            for p_now in point2_int:
                show_image2 = cv2.circle(show_image2, (p_now[0], p_now[1]), radius=draw_radius,
                                         color=(0, 0, 255), thickness=-1)
            image_write = np.concatenate((show_image1, white_image, show_image2), axis=1)
        else:
            show_image1 = None
            show_image2 = None
            image_write = None

        point_ratio1 = np.array([[shape_ori_1[1] / self.image_col,
                                  shape_ori_1[0] / self.image_row]])
        point_ratio2 = np.array([[shape_ori_2[1] / self.image_col,
                                  shape_ori_2[0] / self.image_row]])
        point1 = point1 * point_ratio1
        point2 = point2 * point_ratio2

        # 统计点在网格划分中的分布情况
        num_ratio1 = self.grid_num_large_ratio(shape_ori_1[0], shape_ori_1[1], point1)
        num_ratio2 = self.grid_num_large_ratio(shape_ori_2[0], shape_ori_2[1], point2)
        num_ratio_detect = (num_ratio1 + num_ratio2) / 2

        if point1_num < 1 or point2_num < 1:
            return {'correctness': 0,
                    'homo_dist': -1,
                    'correctness_std': 0,
                    'homo_dist_std': 0,
                    'homography': None,
                    'point1_matched': 0,
                    'point2_matched': 0,
                    'inlier': None,
                    'image_write': image_write,
                    'num_ratio_detect': num_ratio_detect,
                    'num_ratio_match': 0,
                    'num_ratio_ransac': 0}

        # Match the keypoints with the warped_keypoints with nearest neighbor search
        if self.need_HAMMING:
            feature1 = feature1.astype(np.uint8)
            feature2 = feature2.astype(np.uint8)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = bf.match(feature1, feature2)
        matches_idx = np.array([m.queryIdx for m in matches])
        point1_matched = point1[matches_idx, :]
        matches_idx = np.array([m.trainIdx for m in matches])
        point2_matched = point2[matches_idx, :]
        point_match_num = point1_matched.shape[0]

        if point_match_num > 0:
            num_ratio_match1 = self.grid_num_large_ratio(shape_ori_1[0], shape_ori_1[1], point1_matched)
            num_ratio_match2 = self.grid_num_large_ratio(shape_ori_2[0], shape_ori_2[1], point2_matched)
            num_ratio_match = (num_ratio_match1 + num_ratio_match2) / 2
        else:
            num_ratio_match = 0

        # Estimate the homography between the matches using RANSAC
        try_num = self.try_num
        mean_dist_vec = np.zeros((try_num), dtype='float')
        correctness_vec = np.zeros((try_num), dtype='float')
        H = None
        inliers = np.zeros((point1_matched.shape[0])) > 1
        inliers_num = 0
        num_ratio_ransac = 0
        for try_id in range(try_num):
            random_pos = np.random.permutation(point_match_num)
            point1_matched_now = point1_matched[random_pos, :]
            point2_matched_now = point2_matched[random_pos, :]

            H_now, inliers_now = cv2.findHomography(point1_matched_now,
                                                    point2_matched_now, cv2.RANSAC)
            # H_now, inliers_now = cv2.findHomography(point1_matched_now,
            #                                         point2_matched_now, cv2.RHO)
            inliers_now = inliers_now.flatten().astype('bool')

            if H_now is None:
                mean_dist_vec[try_id] = 100
                correctness_vec[try_id] = 0
                continue
            inliers_now_temp = np.zeros((point1_matched.shape[0])) > 1
            inliers_now_temp[random_pos] = inliers_now
            inliers_now = inliers_now_temp

            point1_inlier = point1_matched[inliers_now, :]
            point2_inlier = point2_matched[inliers_now, :]
            if point1_inlier.size > 0:
                num_ratio_ransac1 = self.grid_num_large_ratio(shape_ori_1[0], shape_ori_1[1], point1_inlier)
                num_ratio_ransac2 = self.grid_num_large_ratio(shape_ori_2[0], shape_ori_2[1], point2_inlier)
                num_ratio_ransac = (num_ratio_ransac1 + num_ratio_ransac2) / 2
            else:
                num_ratio_ransac = 0

            inliers_num_now = sum(inliers_now)
            if inliers_num_now > inliers_num:
                H = H_now
                inliers = inliers_now
                inliers_num = inliers_num_now

            corners = np.array([[0, 0, 1],
                                [0, shape_ori_1[0] - 1, 1],
                                [shape_ori_1[1] - 1, 0, 1],
                                [shape_ori_1[1] - 1, shape_ori_1[0] - 1, 1]])
            real_warped_corners = np.dot(corners, np.transpose(real_H))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            real_warped_corners /= point_ratio1
            warped_corners = np.dot(corners, np.transpose(H_now))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            warped_corners /= point_ratio1
            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            correctness = float(mean_dist <= self.soft_dist_homo)

            mean_dist_vec[try_id] = mean_dist
            correctness_vec[try_id] = correctness

        # 是否需要绘制匹配结果
        if write_match_mark:
            point1_homo = point1_matched[inliers, :] / point_ratio1
            point2_homo = point2_matched[inliers, :] / point_ratio2
            point1_homo = np.round(point1_homo).astype('int')
            point2_homo = np.round(point2_homo).astype('int')
            for p_now in point1_homo:
                show_image1 = cv2.circle(show_image1, (p_now[0], p_now[1]), radius=draw_radius,
                                         color=(255, 0, 0), thickness=-1)
            for p_now in point2_homo:
                show_image2 = cv2.circle(show_image2, (p_now[0], p_now[1]), radius=draw_radius,
                                         color=(255, 0, 0), thickness=-1)
            image_write = np.concatenate((show_image1, white_image, show_image2), axis=1)
            for line_id in range(point1_homo.shape[0]):
                p1 = (point1_homo[line_id, 0], point1_homo[line_id, 1])
                p2 = (point2_homo[line_id, 0] + show_image1.shape[1] + draw_margin, point2_homo[line_id, 1])
                show_image2 = cv2.line(image_write, p1, p2, color=(0, 255, 0), thickness=1)

        if H is None:
            return {'correctness': 0,
                    'homo_dist': -1,
                    'correctness_std': 0,
                    'homo_dist_std': 0,
                    'homography': H,
                    'point1_matched': point1_matched / point_ratio1,
                    'point2_matched': point2_matched / point_ratio2,
                    'inlier': inliers,
                    'image_write': image_write,
                    'num_ratio_detect': num_ratio_detect,
                    'num_ratio_match': num_ratio_match,
                    'num_ratio_ransac': num_ratio_ransac}

        correct_mean, correct_std = np.mean(correctness_vec), np.std(correctness_vec)
        dist_mean, dist_std = np.mean(mean_dist_vec), np.std(mean_dist_vec)
        dist_mean_min = np.min(mean_dist_vec)

        return {'correctness': correct_mean,
                'homo_dist': dist_mean,
                'homo_dist_min': dist_mean_min,
                'correctness_std': correct_std,
                'homo_dist_std': dist_std,
                'homography': H,
                'point1_matched': point1_matched / point_ratio1,
                'point2_matched': point2_matched / point_ratio2,
                'inlier': inliers,
                'image_write': image_write,
                'num_ratio_detect': num_ratio_detect,
                'num_ratio_match': num_ratio_match,
                'num_ratio_ransac': num_ratio_ransac}

    def get_homograghy_esti(self, write_match_mark=False,
                            image_name=None, image_ori: np.ndarray = None):
        repeat_num, point_num = self.get_repeat()

        # 每对图像之间，计算单应矩阵
        assert (self.each_num >= 2)
        assert len(self.desc_list) == self.each_num
        # 未设置保存路径，则不能绘制匹配结果
        if self.match_image_path is None:
            write_match_mark = False
        homo_corr_sum = 0
        match_score_sum = 0
        true_num_sum = 0
        homo_corr_std_sum = 0
        num_ratio_detect_sum = 0
        num_ratio_match_sum = 0
        num_ratio_ransac_sum = 0
        pair_num = 0
        for id_1 in range(self.each_num):
            for id_2 in range(id_1 + 1, self.each_num):
                H_1_ref = np.linalg.inv(self.H[id_1])
                H_ref_2 = self.H[id_2]
                H_1_2 = np.dot(H_ref_2, H_1_ref)
                shape_ori_1 = self.image_shape_ori[id_1]
                shape_ori_2 = self.image_shape_ori[id_2]
                # repeat_now, point_num_now = self.get_pair_repeat(self.point_list[id_1],
                #                                                  self.point_list[id_2],
                #                                                  shape_ori_1, shape_ori_2, H_1_2)
                match_score_now, true_num_now = self.get_pair_match_score(
                    self.point_list[id_1], self.point_list[id_2],
                    shape_ori_1, shape_ori_2,
                    self.desc_list[id_1], self.desc_list[id_2], H_1_2)
                homo_result = self.get_pair_homography(self.point_list[id_1], self.point_list[id_2],
                                                       shape_ori_1, shape_ori_2,
                                                       self.desc_list[id_1], self.desc_list[id_2],
                                                       H_1_2,
                                                       write_match_mark,
                                                       image_ori[id_1], image_ori[id_2])

                # 若需要绘制结果，则保存之
                if write_match_mark:
                    path_now = os.path.join(self.match_image_path, image_name)
                    if not os.path.exists(path_now):
                        os.mkdir(path_now)
                    image_write = homo_result['image_write']
                    image_write_name = os.path.join(path_now,
                                                    '%d_%d-corr_%d-dist_%.3f_%.4f-score_%.3f.png' %
                                                    (id_1, id_2,
                                                     round(homo_result['correctness']),
                                                     homo_result['homo_dist_min'],
                                                     homo_result['homo_dist_std'],
                                                     match_score_now))
                    plt.imsave(image_write_name, image_write)

                # 若match_score_now为无效值，说明当前图像对不存在有效的公共区域
                if match_score_now >= 0:
                    homo_corr_sum += homo_result['correctness']
                    match_score_sum += match_score_now
                    true_num_sum += true_num_now
                    homo_corr_std_sum += homo_result['correctness_std']
                    num_ratio_detect_sum += homo_result['num_ratio_detect']
                    num_ratio_match_sum += homo_result['num_ratio_match']
                    num_ratio_ransac_sum += homo_result['num_ratio_ransac']
                    pair_num += 1

        if pair_num < 1:
            return np.zeros((5), dtype='float32')
        homo_corr = homo_corr_sum / pair_num
        match_score = match_score_sum / pair_num
        true_num = true_num_sum / pair_num
        homo_corr_std = homo_corr_std_sum / pair_num
        num_ratio_detect = num_ratio_detect_sum / pair_num
        num_ratio_match = num_ratio_match_sum / pair_num
        num_ratio_ransac = num_ratio_ransac_sum / pair_num
        result_vec = np.array([repeat_num, homo_corr, match_score,
                               homo_corr_std, point_num, true_num,
                               num_ratio_detect, num_ratio_match,
                               num_ratio_ransac], dtype='float32')
        return result_vec
