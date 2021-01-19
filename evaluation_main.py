import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import time
import os
from point_select_utils import remove_out_point
from POP_net_class import POPNet
from demo_superpoint import SuperPointNet
from torch.utils.data import DataLoader
from dataset_class import HPatchesDataset
from evaluation_class import EvaluationSeq

# compute the performance metrics for a methods on a dataset

class EvaluationMain():
    def __init__(self, device_str='cuda:0'):
        self.methed_name = ''
        self.root_dir = ''
        self.resp_thre = -1
        self.nms_rad = -1
        self.max_point_num = -1
        self.soft_dist = -1
        self.out_dist = -1
        self.image_row, self.image_col = -1, -1
        self.xy_range = np.array([-1, -2, -1, -2])
        self.eval_match = False
        self.device = torch.device(device_str)
        self.match_image_path = None

        # SIFT and ORB directly use the implementation of OpenCV
        self.tradition_name = {'SIFT', 'ORB'}
        # ORB use the HAMMING distance
        self.need_HAMMING_name = {'ORB'}

    def set_dataset(self, root_dir: str, match_image_path: str = None):
        self.root_dir = root_dir
        self.match_image_path = match_image_path
        if self.match_image_path is not None:
            if not os.path.exists(match_image_path):
                os.makedirs(match_image_path)

    def set_hyper(self, nms_rad: int, soft_dist: int, out_dist: int,
                  max_point_num: int, resp_thre: float,
                  image_row: int, image_col: int):
        if self.resp_thre is not None:
            self.resp_thre = resp_thre
        self.nms_rad = nms_rad
        self.max_point_num = max_point_num
        self.soft_dist = soft_dist
        self.out_dist = out_dist
        self.image_row, self.image_col = image_row, image_col
        self.xy_range = np.array([out_dist, self.image_col - out_dist,
                                  out_dist, self.image_row - out_dist])

    def main(self, methed_name: str, para_dict: dict = None):
        # record the runtime
        time_num_list = [0, 0]

        self.methed_name = methed_name
        device = self.device
        self.eval_match = ('eval_match' in para_dict.keys() and
                           para_dict['eval_match'])

        large_point_num = para_dict['large_point_num'] if \
            ('large_point_num' in para_dict.keys()) else self.max_point_num

        data_HPatches = HPatchesDataset(self.root_dir,
                                        self.image_row, self.image_col, 'total')
        dataloader = DataLoader(data_HPatches, batch_size=1, shuffle=False)

        if self.methed_name == 'POP':
            if 'desc_len' in para_dict.keys():
                POP_net = POPNet(para_dict['desc_len'])
            else:
                POP_net = POPNet()
            checkpoint = torch.load(para_dict['our_model_name'], map_location=device)
            POP_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            POP_net.eval()
            POP_net.to(device)
        elif self.methed_name == 'superpoint':
            # Superpoint
            superpoint_net = SuperPointNet()
            superpoint_net.load_state_dict(torch.load('superpoint_v1.pth'))
            superpoint_net.eval()
            superpoint_net.to(device)
        elif self.methed_name == 'SIFT':
            point_obj = cv2.xfeatures2d.SIFT_create(nfeatures=self.max_point_num)
            # point_obj = cv2.SIFT_create(nfeatures=self.max_point_num)
        elif self.methed_name == 'ORB':
            point_obj = cv2.ORB_create(nfeatures=self.max_point_num * 5)

        def get_POP_point(input_):
            time_cum = 0

            input_ = input_.to(device)
            with torch.no_grad():
                time_last = time.time()
                score_output, desc = POP_net(input_)
                # transform to probability
                score = torch.sigmoid(score_output).detach()
                time_cum += (time.time() - time_last)

                desc = desc.detach()

            time_last = time.time()
            point_list_res = self.get_point_from_resp(score)
            desc_list_res = self.get_easy_desc_from_map(point_list_res, desc)
            time_cum += (time.time() - time_last)

            # record the runtime
            time_num_list[0] += time_cum
            time_num_list[1] += len(point_list_res)
            return point_list_res, desc_list_res

        def get_superpoint_point(input_gray_):
            image_num_ = input_gray_.shape[0]
            input_gray_ = input_gray_.to(device)

            time_last = time.time()

            with torch.no_grad():
                semi, coarse_desc = superpoint_net(input_gray_)
                coarse_desc = coarse_desc.detach()
                semi = semi.cpu().numpy()

            output_resp = torch.zeros(image_num_, 1, self.image_row, self.image_col)
            for image_id_ in range(image_num_):
                # --- Process points.
                dense = np.exp(semi[image_id_])  # Softmax.
                dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
                # Remove dustbin.
                nodust = dense[:-1, :, :]
                # Reshape to get full resolution heatmap.
                cell_size = 8
                image_height, image_width = self.image_row, self.image_col
                Hc = int(image_height / cell_size)
                Wc = int(image_width / cell_size)
                nodust = nodust.transpose(1, 2, 0)
                heatmap = np.reshape(nodust, [Hc, Wc, cell_size, cell_size])
                heatmap = np.transpose(heatmap, [0, 2, 1, 3])
                heatmap = np.reshape(heatmap, [Hc * cell_size, Wc * cell_size])
                heatmap = torch.from_numpy(heatmap).to(device)
                output_resp[image_id_][0] = heatmap

            point_list_res = self.get_point_from_resp(output_resp)
            desc_list_res = self.calcu_desc_from_map(point_list_res, coarse_desc)

            time_cum = (time.time() - time_last)

            # record the runtime
            time_num_list[0] += time_cum
            time_num_list[1] += len(point_list_res)

            return point_list_res, desc_list_res

        def get_tradition_point(image_ori_):
            image_num_ = image_ori_.shape[0]
            point_list_res = []
            desc_list_res = []

            time_cum = 0
            for image_id_ in range(image_num_):
                image_now = image_ori_[image_id_]
                gray = cv2.cvtColor(image_now, cv2.COLOR_RGB2GRAY)

                time_last = time.time()
                kp = point_obj.detect(gray, None)
                _, desc_now = point_obj.compute(gray, kp)
                time_cum += (time.time() - time_last)

                point_now = np.array([p.pt for p in kp], dtype='float32')
                point_resp = np.array([p.response for p in kp], dtype='float32')
                # add the confidence into the third column
                point_now = np.c_[point_now, point_resp]
                if point_now.size > 1:
                    point_now, remain_ind = self.select_k_best_self(point_now)
                    desc_now = desc_now[remain_ind, :]

                point_list_res.append(point_now)
                desc_list_res.append(desc_now)

            # record the runtime
            time_num_list[0] += time_cum
            time_num_list[1] += len(point_list_res)
            return point_list_res, desc_list_res

        # ##### the definitions of the subfunctions of algorithms are finished ######
        result_item = ['repeat', 'homo_corr', 'm_score', 'homo_corr_std',
                       'point_num', 'true_num', 'num_ratio_detect',
                       'num_ratio_match', 'num_ratio_ransac']
        result_mat = np.zeros((len(dataloader), len(result_item)), dtype='float')
        for idx, batch in enumerate(dataloader, 0):
            # obtain the image and groundtruth
            input, H, image_shape, image_name = batch['image'], batch['H'], \
                                                batch['image_shape'], batch['image_name']
            input_gray = batch['image_gray']
            image_ori = batch['image_ori']
            batch_size = input.size(0)
            input_each_num = input.size(1)
            assert batch_size==1
            # reshape the data as 4-D tensors
            input = input.view(tuple(np.r_[input.shape[0] * input.shape[1],
                                           input.shape[2:]]))
            H = H.view(tuple(np.r_[H.shape[0] * H.shape[1], H.shape[2:]]))
            image_shape_ori = image_shape.view(tuple(np.r_[image_shape.shape[0] * image_shape.shape[1],
                                                           image_shape.shape[2:]]))
            input_gray = input_gray.view(tuple(np.r_[input_gray.shape[0] * input_gray.shape[1],
                                                     input_gray.shape[2:]]))
            image_ori = image_ori.view(tuple(np.r_[image_ori.shape[0] * image_ori.shape[1],
                                                   image_ori.shape[2:]]))
            H = H.numpy()
            image_shape_ori = image_shape_ori.numpy()
            image_ori = image_ori.numpy()

            if self.methed_name == 'superpoint':
                point_list, desc_list = get_superpoint_point(input_gray)
            elif self.methed_name in self.tradition_name:
                point_list, desc_list = get_tradition_point(image_ori)
            elif self.methed_name == 'POP':
                point_list, desc_list = get_POP_point(input)
            else:
                point_list = None
                desc_list = None
                assert 'unknown method'

            if self.methed_name in self.need_HAMMING_name:
                need_HAMMING = True
            else:
                need_HAMMING = False

            eval_obj = EvaluationSeq(point_list, desc_list, H,
                                     self.soft_dist, self.out_dist,
                                     self.image_row, self.image_col, image_shape_ori,
                                     need_HAMMING, self.match_image_path)

            result_vec = eval_obj.get_homograghy_esti(
                write_match_mark=True, image_name=image_name[0],
                image_ori=image_ori)

            result_mat[idx] = result_vec

            print('(epsilon=%d)' % self.soft_dist, 'id:%d' % idx,
                  '%s: %.3f' % (result_item[0], result_vec[0]),
                  '%s: %.3f' % (result_item[1], result_vec[1]),
                  '%s: %.3f' % (result_item[2], result_vec[2]),
                  '%s: %.3f' % (result_item[3], result_vec[3]),
                  '%s: %.3f' % (result_item[4], result_vec[4]))

        result_mean = np.mean(result_mat, axis=0)
        print('## accuracy on the entire dataset (epsilon=%d): \n ## ' % self.soft_dist,
              '%s: %.5f' % (result_item[0], result_mean[0]),
              '%s: %.5f' % (result_item[1], result_mean[1]),
              '%s: %.5f' % (result_item[2], result_mean[2]),
              '%s: %.5f' % (result_item[3], result_mean[3]),
              '%s: %.5f' % (result_item[4], result_mean[4]))

        time_mean = time_num_list[0] / (time_num_list[1] + 1e-10)
        return result_mean, result_mat, result_item, time_mean

    def get_point_from_resp(self, resp_tensor):
        resp_max = F.max_pool2d(resp_tensor, kernel_size=2 * self.nms_rad + 1,
                                padding=self.nms_rad, stride=1)
        max_pos_tensor = (resp_max == resp_tensor).transpose(1, 0)
        max_pos_tensor = max_pos_tensor.squeeze(0)

        input_each_num, image_row, image_col = \
            resp_tensor.shape[0], resp_tensor.shape[2], resp_tensor.shape[3]
        xy_range = np.array([self.out_dist, image_col - self.out_dist,
                             self.out_dist, image_row - self.out_dist])

        point_list = []
        for image_id in range(input_each_num):
            resp_now = resp_tensor[image_id].cpu().numpy().squeeze()

            max_pos_now = max_pos_tensor[image_id].cpu().numpy().squeeze().astype('bool')
            max_mask = (max_pos_now & (resp_now > self.resp_thre))
            point_y, point_x = np.where(max_mask)
            point_here = remove_out_point(np.c_[point_x, point_y], xy_range)
            point_loc = point_here[:, 1] * image_col + point_here[:, 0]
            point_resp = resp_now.take(point_loc)
            point_now = np.c_[point_here[:, 0], point_here[:, 1], point_resp]
            point_now, point_ind = self.nms_fast(point_now, self.nms_rad,
                                                 (image_row, image_col))
            point_now, _ = self.select_k_best(point_now, self.max_point_num)
            point_list.append(point_now)

        return point_list

    def calcu_desc_from_map(self, point_list: list, coarse_desc: torch.Tensor):
        image_num = len(point_list)
        desc_list = []
        for image_id in range(image_num):
            coarse_desc_now = coarse_desc[image_id]
            coarse_desc_now = coarse_desc_now.view(tuple(np.r_[1, coarse_desc_now.shape[:]]))
            point_now = point_list[image_id]
            point_num = point_now.shape[0]
            if point_num < 1:
                desc_list.append(np.zeros(0))
                continue

            samp_pts = torch.from_numpy(point_now[:, :2].copy())
            samp_pts[:, 0] = (samp_pts[:, 0] / (self.image_col / 2.)) - 1.
            samp_pts[:, 1] = (samp_pts[:, 1] / (self.image_row / 2.)) - 1.
            samp_pts = samp_pts.contiguous().view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(coarse_desc.device)
            desc = torch.nn.functional.grid_sample(coarse_desc_now, samp_pts)
            desc = desc.data.cpu().numpy().reshape(-1, point_num)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            desc = desc.transpose((1, 0))
            desc_list.append(desc)

        return desc_list

    def get_easy_desc_from_map(self, point_list: list, desc: torch.Tensor):
        image_num = len(point_list)
        desc_list = []
        for image_id in range(image_num):
            point_now = point_list[image_id]
            point_num = point_now.shape[0]
            if point_num < 1:
                desc_list.append(np.zeros(0))
                continue

            point_desc = desc[image_id, :, point_now[:, 1],
                         point_now[:, 0]].transpose(1, 0)

            desc_list.append(point_desc.cpu().numpy())

        return desc_list

    def select_k_best(self, points: np.ndarray, k: int):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        if points.size < 1:
            return points, np.zeros(0)
        sort_ind = points[:, 2].argsort()
        sorted_prob = points[sort_ind, :]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :], sort_ind[-start:]

    def nms_fast(self, points: np.ndarray, dist_thresh: int, image_shape: tuple = None):
        if image_shape is None:
            image_shape = (self.image_row, self.image_col)
        image_row, image_col = image_shape[:]
        grid = np.zeros((image_row, image_col)).astype(int)  # Track NMS data.
        inds = np.zeros((image_row, image_col)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-points[:, 2])
        points = points[inds1, :]
        rpoints = points[:, :2].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rpoints.shape[0] == 0:
            return np.zeros((0, 3)).astype(int), np.zeros(0).astype(int)
        if rpoints.shape[0] == 1:
            out = np.c_[rpoints, points[:, 2]].reshape(1, 3)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rpoints):
            grid[rpoints[i, 1], rpoints[i, 0]] = 1
            inds[rpoints[i, 1], rpoints[i, 0]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rpoints):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = points[inds_keep, :]
        values = out[:, -1]
        inds2 = np.argsort(-values)
        out = out[inds2, :]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def select_k_best_self(self, point_ori):
        return self.select_k_best(point_ori, self.max_point_num)
