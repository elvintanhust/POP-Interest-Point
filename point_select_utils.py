import numpy as np
import cv2
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_dilation

# several utility functions for the selection of interest point

def get_circle_win(rad: int, dtype: str):
    win_size = 2 * rad + 1
    win = np.zeros((win_size, win_size), dtype=dtype)
    c_mat = np.tile(np.arange(-rad, rad + 1, dtype='int'), (win_size, 1))
    r_mat = c_mat.transpose((1, 0))
    win[(r_mat * r_mat + c_mat * c_mat) <= rad * rad] = 1
    return win


def get_circle_win_tensor(rad: int, dtype: str):
    win_size = 2 * rad + 1
    win = torch.zeros((win_size, win_size), dtype=dtype)
    c_mat = np.tile(np.arange(-rad, rad + 1, dtype='int'), (win_size, 1))
    r_mat = c_mat.transpose((1, 0))
    fore_pos = ((r_mat * r_mat + c_mat * c_mat) <= rad * rad)
    win.view(-1)[np.where(fore_pos.reshape(-1))] = 1
    return win


def remove_out_point(point: np.ndarray, xy_range: np.ndarray):
    if point.size < 1:
        return point
    point_result = point[(point[:, 0] > xy_range[0]) &
                         (point[:, 0] < xy_range[1]) &
                         (point[:, 1] > xy_range[2]) &
                         (point[:, 1] < xy_range[3]), :]
    return point_result


def get_mean_repeat(output_resp: torch.Tensor, H: torch.Tensor,
                    local_rad: int, soft_dist: int, out_dist: int,
                    min_need_avail_num: int):
    each_num = output_resp.shape[0]
    # obtain the local maximum
    output_resp_max = F.max_pool2d(output_resp, kernel_size=2 * local_rad + 1,
                                   padding=local_rad, stride=1)
    local_win = get_circle_win_tensor(local_rad, dtype=output_resp.dtype)
    local_win = local_win.to(output_resp.device)
    local_win_kernel = local_win.repeat((each_num, 1, 1, 1))
    max_pos_tensor = (output_resp_max == output_resp).transpose(1, 0)
    max_num_tensor = F.conv2d(max_pos_tensor.to(torch.float32),
                              local_win_kernel,
                              padding=local_rad, stride=1, groups=each_num)
    max_pos_tensor = max_pos_tensor.squeeze(0)
    max_num_tensor = max_num_tensor.squeeze(0)

    image_row, image_col = output_resp.shape[-2:]
    soft_win = get_circle_win(soft_dist, dtype='bool')
    point_cum = np.zeros((image_row, image_col), dtype='float32')
    bw_ref_vec = np.zeros((each_num, image_row, image_col), dtype='bool')
    xy_range = np.array([out_dist, image_col - out_dist,
                         out_dist, image_row - out_dist])
    resp_thre = 0.5

    # avail_cum store the visible number of every point
    back_resp_value = -3
    avail_cum = np.zeros([image_row, image_col], dtype='int')
    fore_num_cum_runtime = 0

    # transform every image to the reference image
    for image_id in range(each_num):
        resp = output_resp[image_id].cpu().numpy().squeeze()
        # the generated homography matrix
        h_now = H[image_id].numpy()
        h_inv_now = np.linalg.inv(h_now)
        # cumulate the visible point
        mat_ref = cv2.warpPerspective(resp.astype('int'), h_inv_now,
                                      (image_col, image_row),
                                      borderValue=back_resp_value,
                                      flags=cv2.INTER_NEAREST)
        avail_pos_now = (mat_ref > (back_resp_value + 1))
        avail_cum[avail_pos_now] += 1

        max_pos_now = max_pos_tensor[image_id].cpu().numpy().squeeze().astype('bool')
        # guarantee at most one point can be selected as interest point in a local window
        max_mask = (max_pos_now & (resp > resp_thre))
        max_num_mat = np.round(max_num_tensor[image_id].cpu().numpy().squeeze())
        max_mask = (max_mask & (max_num_mat == 1))
        point_y, point_x = np.where(max_mask)

        # record the number of interest point
        fore_num_cum_runtime += point_y.size

        if point_y.size < 1:
            continue

        point_here = remove_out_point(np.c_[point_x, point_y], xy_range)
        if point_here.size < 1:
            continue

        point_here = point_here.astype('float32')[np.newaxis, :]
        point_ref = cv2.perspectiveTransform(point_here, h_inv_now)
        point_ref = np.rint(point_ref).astype('int').squeeze(0)
        point_ref = remove_out_point(point_ref, xy_range)
        if point_here.size < 1:
            continue

        ref_loc = point_ref[:, 1] * image_col + point_ref[:, 0]
        bw_ref = np.zeros((image_row, image_col), dtype='bool')
        bw_ref.put(ref_loc, True)
        bw_ref = binary_dilation(bw_ref, soft_win)
        bw_ref_vec[image_id] = bw_ref

        point_cum[bw_ref] += 1

    # 考虑到投影变换后无效区域会导致重复数目统计不完整，借助像素的有效次数进行弥补
    # 有效次数多于阈值的，弥补其数值，否则将其重复数目视为0
    # 视为0的意义在于，这些点不会因为投影至无效区域而被优先选为负样本
    need_improve_pos = (avail_cum >= min_need_avail_num) & (avail_cum < each_num)
    point_cum[need_improve_pos] = (point_cum[need_improve_pos] *
                                   (each_num / avail_cum[need_improve_pos]))
    point_cum[avail_cum < min_need_avail_num] = 0

    avail_pos = avail_cum >= min_need_avail_num

    final_soft_rad = soft_dist
    final_soft_win = get_circle_win(final_soft_rad, dtype='float32')
    point_cum_focus = point_cum.copy()
    point_cum_focus = np.rint(point_cum_focus).astype('int')
    point_cum = np.rint(point_cum).astype('int')
    # obtain the maximum
    point_cum_max = maximum_filter(point_cum_focus, footprint=final_soft_win)
    point_cum[np.logical_not(point_cum_focus == point_cum_max)] = 0
    cum_max_pos = (point_cum > 0)
    repeat_num_sum = 0
    point_num_sum = 0
    for image_id in range(each_num):
        bw_now = (bw_ref_vec[image_id] & avail_pos & cum_max_pos)
        repeat_now = point_cum[bw_now]
        if repeat_now.size < 1:
            continue
        repeat_num_sum += np.sum(repeat_now)
        point_num_sum += repeat_now.size

    if point_num_sum < 1:
        return 0, point_cum, fore_num_cum_runtime
    repeat_mean = repeat_num_sum / point_num_sum
    return repeat_mean, point_cum, fore_num_cum_runtime


def calcu_desc_dist(desc: torch.Tensor, H: torch.Tensor, range_now: range,
                    repeat_x_ref: np.ndarray, repeat_y_ref: np.ndarray,
                    xy_range: np.ndarray, dist_margin: dict, pair_thre_init: dict,
                    disc_corr_weight: float, local_rad: int):
    # 当need_grad=False时，计算真实的距离值
    # 但need_grad=True时，达到限定值的距离将设为固定值，从而视为无损失

    input_each_num = len(range_now)
    feature_len = desc.shape[1]

    repeat_xy_ref = np.c_[repeat_x_ref, repeat_y_ref][np.newaxis, :].astype('float32')
    candidate_num = repeat_y_ref.size
    candidate_id = np.arange(candidate_num)
    # # 计算候选点两两之间的距离，距离过近的点不计算cross距离
    # xy_ref1 = np.expand_dims(np.squeeze(repeat_xy_ref, 0), 1)
    # xy_ref0 = np.expand_dims(np.squeeze(repeat_xy_ref, 0), 0)
    # dist_cross = np.linalg.norm(xy_ref1 - xy_ref0,
    #                             ord=None, axis=2)
    # dist_cross = torch.from_numpy(dist_cross).to(device=desc.device)
    # small_cross_mask = dist_cross < (local_rad/2+0.1)
    # 得到候选点在原图像中的位置，并获取相应位置的特征向量
    # 考虑到可能存在位于图像范围外的点，同时记录其是否位于图像范围内
    feature_candi = torch.zeros((input_each_num, candidate_num, feature_len),
                                dtype=desc.dtype, device=desc.device)
    inner_label = torch.zeros((input_each_num, candidate_num),
                              dtype=desc.dtype, device=desc.device)
    for local_id, image_id in enumerate(range_now):
        h_now = H[image_id].numpy()
        point_now = cv2.perspectiveTransform(repeat_xy_ref, h_now)
        point_now = np.rint(point_now).astype('int').squeeze(0)
        point_now = np.c_[point_now, candidate_id]
        point_now = remove_out_point(point_now, xy_range)
        feature_candi[local_id, point_now[:, 2], :] = desc[image_id, :, point_now[:, 1],
                                                      point_now[:, 0]].transpose(1, 0)
        inner_label[local_id, point_now[:, 2]] = 1
    # 定义参与计算特征距离的图像对
    image_pair = np.c_[np.arange(input_each_num), np.arange(1, input_each_num + 1)]
    image_pair[input_each_num - 1, 1] = 0
    corr_target = dist_margin['corr_target']
    cross_target = dist_margin['cross_target']
    corr_margin = dist_margin['corr_margin']
    cross_margin = dist_margin['cross_margin']

    # 计算对应位置的特征距离
    need_posi_pair_num = pair_thre_init['need_posi_num']
    ignore_value_corr = -1000
    corr_similar = torch.sum(feature_candi[image_pair[:, 0]] * feature_candi[image_pair[:, 1]], dim=2)
    corr_exist_mask = inner_label[image_pair[:, 0]] * inner_label[image_pair[:, 1]]
    # 点对存在且距离已满足margin要求，直接调整为理想距离
    corr_similar[(corr_similar.detach() >= corr_margin) & (corr_exist_mask > 0.5)] = corr_margin
    corr_similar = torch.sum(corr_similar, dim=0)
    corr_exist_num = torch.sum(corr_exist_mask, dim=0)
    corr_similar[corr_exist_num < need_posi_pair_num] = ignore_value_corr
    corr_exist_num = corr_exist_num.clamp(min=1)
    corr_similar /= corr_exist_num

    # 计算非对应位置的特征距离
    need_neg_pair_num_in = pair_thre_init['need_neg_num_in']
    need_neg_pair_num_out = pair_thre_init['need_neg_num_out']
    ignore_value_cross = candidate_num * 2
    cross_similar = torch.matmul(feature_candi[image_pair[:, 0]],
                                 feature_candi[image_pair[:, 1]].transpose(1, 2))
    cross_exist_mask = torch.matmul(
        inner_label[image_pair[:, 0]].view((input_each_num, candidate_num, 1)),
        inner_label[image_pair[:, 1]].view((input_each_num, 1, candidate_num)))
    # 点对存在且距离已满足margin要求，直接调整为理想距离
    # cross_similar[((cross_similar.detach() <= cross_margin) & (cross_exist_mask > 0.5)) |
    #               small_cross_mask] = cross_margin
    cross_similar[(cross_similar.detach() <= cross_margin) & (cross_exist_mask > 0.5)] = cross_margin
    cross_similar = torch.sum(cross_similar, dim=0)
    # 得到存在的点对位置
    cross_exist_num = torch.sum(cross_exist_mask, dim=0)
    cross_exist_mask = cross_exist_num >= need_neg_pair_num_in

    # 对角线对应的图像对是对应像对，应当去除
    cross_exist_mask[candidate_id, candidate_id] = 0
    cross_similar[cross_exist_num < need_neg_pair_num_in] = 0
    cross_similar[candidate_id, candidate_id] = 0
    cross_exist_num = cross_exist_num.clamp(min=1)
    cross_similar /= cross_exist_num
    # 对于每个点，求取其与其他点的特征距离均值，要求点对数目高于阈值要求
    cross_exist_num = torch.sum(cross_exist_mask, dim=1).to(torch.float32)
    cross_similar = torch.sum(cross_similar, dim=1)
    cross_similar[cross_exist_num < need_neg_pair_num_out] = ignore_value_cross
    cross_exist_num = cross_exist_num.clamp(min=1)
    cross_similar /= cross_exist_num

    # 计算可分性
    point_disc = disc_corr_weight * corr_similar - (1 - disc_corr_weight) * cross_similar

    return point_disc


def get_multi_patch(image: torch.Tensor, r_start, c_start, r_len, c_len):
    image_channel, image_row, image_col = image.shape[-3:]

    r = r_start[:, np.newaxis, np.newaxis]
    dr = np.arange(r_len)[np.newaxis, :, np.newaxis]
    c = c_start[:, np.newaxis, np.newaxis]
    dc = np.arange(c_len)[np.newaxis, np.newaxis, :]
    ind = (r + dr) * image_col + (c + dc)
    # 加入通道
    ind = ind[:, np.newaxis, :, :]
    dl = np.arange(image_channel)[np.newaxis, :, np.newaxis, np.newaxis]
    ind = ind + dl * (image_col * image_row)

    ind_shape_ori = ind.shape
    ind = ind.reshape(-1)
    patches = image.view(-1)[ind]
    patches = patches.reshape(ind_shape_ori)
    return patches


def calcu_info(desc: torch.Tensor, H: torch.Tensor, range_now: range,
               repeat_x_ref: np.ndarray, repeat_y_ref: np.ndarray,
               xy_range: np.ndarray, out_dist: int,
               recon_target, net_recon, need_grad):
    patch_size = 16
    patch_rad = math.ceil(patch_size / 2)
    dist_d = max(0, patch_rad - out_dist + 1)
    xy_range = np.array([xy_range[0] + dist_d, xy_range[1] - dist_d,
                         xy_range[2] + dist_d, xy_range[3] - dist_d])

    input_each_num = len(range_now)

    # 因为计算信息量时对边界补白的要求更高，会除去一部分点
    # 需记录哪些位置的点被保留，从而便于后续回复到输入点的形式
    ref_id = np.arange(repeat_x_ref.shape[0])
    repeat_xy_ref = np.c_[repeat_x_ref, repeat_y_ref, ref_id]
    repeat_xy_ref = remove_out_point(repeat_xy_ref, xy_range)
    keep_ref_id = repeat_xy_ref[:, 2]
    repeat_xy_ref = repeat_xy_ref[:, :2]
    candidate_num = repeat_xy_ref.shape[0]

    # 得到候选点在原图像中的位置，并获取相应位置的特征向量
    # 考虑到可能存在位于图像范围外的点，同时记录其是否位于图像范围内
    repeat_xy_ref = repeat_xy_ref[np.newaxis, :].astype('float32')
    candidate_id = np.arange(candidate_num)
    recon_device = next(net_recon.parameters()).device
    info_mat = torch.zeros((input_each_num, candidate_num),
                           dtype=recon_target.dtype, device=recon_device)
    inner_label = torch.zeros((input_each_num, candidate_num),
                              dtype=torch.bool, device=recon_device)
    loss_info_sum = 0
    loss_info_num = 0
    for local_id, image_id in enumerate(range_now):
        h_now = H[image_id].numpy()
        point_now = cv2.perspectiveTransform(repeat_xy_ref, h_now)
        point_now = np.rint(point_now).astype('int').squeeze(0)
        point_now = np.c_[point_now, candidate_id]
        point_now = remove_out_point(point_now, xy_range)
        # 记录位于图像范围内的特征和点编号
        inner_label_now = torch.zeros((candidate_num,),
                                      dtype=torch.bool, device=desc.device)
        inner_label_now[point_now[:, 2]] = 1
        inner_label[local_id] = inner_label_now
        # 得到特征点附近的patches,即重建的gt
        r_start = point_now[:, 1] - patch_rad
        c_start = point_now[:, 0] - patch_rad
        r_len = patch_size
        c_len = patch_size
        gt_patch = get_multi_patch(recon_target[image_id], r_start, c_start, r_len, c_len)
        # 进行图像重建
        gt_patch = gt_patch.to(recon_device)
        if need_grad:
            patch_recon = net_recon(gt_patch)
        else:
            with torch.no_grad():
                patch_recon = net_recon(gt_patch)
        # # 在内存而不是显存中计算信息量
        # gt_temp = (gt_patch.cpu().numpy().transpose((0, 2, 3, 1)) * 129 + 128).astype('uint8')
        # recon_temp = (patch_recon.cpu().numpy().transpose((0, 2, 3, 1)) * 129 + 128).astype('uint8')

        dist = patch_recon - gt_patch
        dist = dist * dist
        info_vec = torch.mean(torch.mean(torch.mean(dist, dim=3), dim=2), dim=1)
        if need_grad:
            loss_info_sum += torch.sum(info_vec)
            loss_info_num += torch.sum(inner_label_now.int()).item()
        else:
            info_mat[local_id, inner_label_now] = info_vec

    # 若need_grad为False，说明此次计算的结果需要还原到每个输入点
    if need_grad:
        loss_info = loss_info_sum / loss_info_num
        return None, loss_info
    else:
        inner_num = torch.sum(inner_label.float(), dim=0).to(recon_device)
        point_info_keep = torch.sum(info_mat, dim=0) / inner_num.clamp(min=1)
        point_info = torch.zeros((repeat_x_ref.shape[0],),
                                 dtype=info_mat.dtype, device=info_mat.device)
        point_info[keep_ref_id] = point_info_keep
        return point_info, None
