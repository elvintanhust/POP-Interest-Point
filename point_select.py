import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.morphology import binary_dilation

from point_select_utils import get_circle_win, remove_out_point, calcu_desc_dist
from point_select_utils import calcu_info
from point_select_utils import get_mean_repeat


def get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                   back_loss_cum, repeat_mean_cum, image_base_num,
                   fore_num_cum_runtime, image_real_num,
                   disc_ratio, disc_value, info_value):
    return {'fore_num_cum': fore_num_cum,
            'fore_loss_cum': fore_loss_cum,
            'back_num_cum': back_num_cum,
            'back_loss_cum': back_loss_cum,
            'repeat_mean_cum': repeat_mean_cum,
            'image_base_num': image_base_num,
            'fore_num_cum_runtime': fore_num_cum_runtime,
            'image_real_num': image_real_num,
            'disc_ratio': disc_ratio,
            'disc_value': disc_value,
            'info_value': info_value}


# 尝试不允许重建损失对特征点模型进行更新
def point_select(score_output: torch.Tensor, score_detach: torch.Tensor,
                  desc: torch.Tensor, gt: torch.Tensor, H: torch.Tensor,
                  range_now: range, local_rad: int,
                  out_rad: int, inner_rad: int, soft_dist: int,
                  back1_weight: float, out_dist: int, image_ori: torch.Tensor,
                  recon_target: torch.Tensor, net_recon, need_info_mark
                  ):
    # 得到网络输出的非求导副本
    desc_detach = desc.detach()
    gt_vec = gt.view(-1)
    score_detach_vec = score_detach.view(-1)
    MSE_loss = nn.MSELoss()
    image_row, image_col = score_detach.shape[-2:]
    score_flatten = score_detach.view(-1)
    input_each_num = len(range_now)
    # 统计投影变换后每个像素位于有效区域的次数
    back_score_value = -3
    # min_need_avail_num = 5
    min_need_avail_num = 1
    exit_min_point_num = 3

    # 超参数相关
    fore_around_win = get_circle_win(local_rad, dtype='bool')
    xy_range = np.array([out_dist, image_col - out_dist, out_dist, image_row - out_dist])
    # 点数目分布区间
    min_point_num = 200
    max_point_num = 400
    # 正负样本的大概比例
    back1_num_ratio = 0.8
    back2_num_ratio = 0.5
    # 计算可分性中同名点距离对应权重
    disc_corr_weight_min = input_each_num / max_point_num
    # 排序筛选正样本中重复率权重
    # repeat_weight = 1
    repeat_weight = np.random.uniform(0.5, 1)
    # repeat_weight = np.random.uniform(0.3, 0.9)
    if need_info_mark:
        info_weight = np.random.uniform(0, 0.25)
    else:
        info_weight = 0

    # 打印信息相关
    fore_num_cum = 0
    fore_loss_cum = 0
    back_num_cum = 0
    back_loss_cum = 0
    repeat_mean_cum = 0
    image_base_num = 0
    fore_num_cum_runtime = 0
    image_real_num = 0
    disc_value = 0
    disc_ratio = 0
    info_value = 0

    repeat_num_mean, point_cum, fore_num_cum_runtime = \
        get_mean_repeat(score_detach[range_now], H[range_now], local_rad,
                        soft_dist, out_dist, min_need_avail_num)
    image_real_num = input_each_num

    # 记录重复率均值
    repeat_mean_cum += repeat_num_mean
    image_base_num += 1

    # 得到有可能被选为正样本的点，并按照重复率排序
    min_repeat_thre = 1
    y_total, x_total = np.where(point_cum >= min_repeat_thre)
    point_cum_total = point_cum[y_total, x_total]
    cum_total_sort_ind = np.argsort(point_cum_total)[::-1]
    cum_total_sort = point_cum_total[cum_total_sort_ind]
    y_total = y_total[cum_total_sort_ind]
    x_total = x_total[cum_total_sort_ind]
    # 根据重复率，选择排名靠前的点作为候选特征点
    candidate_num_init = min(1000, cum_total_sort.size)
    candidate_num_max = 2000
    candi_repeat_thre = max(cum_total_sort[candidate_num_init - 1], min_repeat_thre)
    candidate_num = np.sum(cum_total_sort >= candi_repeat_thre)

    point_repeat = cum_total_sort[:candidate_num]
    repeat_y_ref = y_total[:candidate_num]
    repeat_x_ref = x_total[:candidate_num]

    # 达到重复率阈值的点作为候选前景
    dist_value_config = {'corr_target': 1, 'cross_target': -1,
                         'corr_margin': 1, 'cross_margin': 0.2}

    # if candidate_num < 3:
    if candidate_num < exit_min_point_num:
        print_dict = get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                                    back_loss_cum, repeat_mean_cum, image_base_num,
                                    fore_num_cum_runtime, image_real_num,
                                    disc_ratio, disc_value, info_value)
        return 0, False, 0, print_dict

    pair_thre_init = {'need_posi_num': 2.5, 'need_neg_num_in': 4.5,
                      'need_neg_num_out': candidate_num / 3}
    # 若当前点过多，则计算可分性将超出显存空间，此时不计算可分性而完全利用重复率筛选正负样本
    if candidate_num <= candidate_num_max:
        # 计算可分性
        point_disc = calcu_desc_dist(desc_detach, H, range_now, repeat_x_ref, repeat_y_ref,
                                     xy_range, dist_value_config, pair_thre_init, disc_corr_weight_min, local_rad)
        # max_disc_value为理论上的最大可分性取值
        max_disc_value = (disc_corr_weight_min * dist_value_config['corr_margin'] -
                          (1 - disc_corr_weight_min) * dist_value_config['cross_margin'])
        # 这里min_disc_value并非理论最小可分性取值
        # 使用理论最小会使得实际得到的可分性取值过于集中，因为实际中几乎不会出现理论最小
        min_disc_value = (disc_corr_weight_min * dist_value_config['cross_margin'] -
                          (1 - disc_corr_weight_min) * dist_value_config['corr_margin'])
        point_disc = (point_disc - min_disc_value) / (max_disc_value - min_disc_value)

        # 此时只评价信息量，不计算梯度
        if need_info_mark:
            point_info, _ = calcu_info(
                desc_detach, H, range_now, repeat_x_ref, repeat_y_ref,
                xy_range, out_dist, recon_target, net_recon, False)
        else:
            point_info = torch.zeros(candidate_num, dtype=desc_detach.dtype, device=desc_detach.device)
    else:
        # 判别性
        point_disc = torch.zeros(candidate_num, dtype=desc_detach.dtype, device=desc_detach.device)
        # 独特性
        point_info = torch.zeros(candidate_num, dtype=desc_detach.dtype, device=desc_detach.device)

    # 调试使用
    # if torch.max(point_disc) > 2:
    #     temp = calcu_desc_dist(desc_detach, H, range_now, repeat_x_ref, repeat_y_ref,
    #                            xy_range, dist_value_config, pair_thre_init)

    # 考虑到某些点投影后位于有效范围外，将有效次数不满足要求的计算结果置为随机无效值
    # 避免无效结果排序后前景点过于集中于某个局部
    out_disc_value = dist_value_config['cross_target'] - dist_value_config['corr_target']
    out_disc_pos = (point_disc <= out_disc_value)
    out_num = torch.sum(out_disc_pos).item()
    point_disc[out_disc_pos] = 10 * out_disc_value - torch.rand(out_num, dtype=desc.dtype, device=desc.device)

    # 按照可分性对点进行排序
    avail_disc_num = torch.sum(point_disc > out_disc_value).item()
    if avail_disc_num < exit_min_point_num:
        print_dict = get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                                    back_loss_cum, repeat_mean_cum, image_base_num,
                                    fore_num_cum_runtime, image_real_num,
                                    disc_ratio, disc_value, info_value)
        return 0, False, 0, print_dict

    # 联合重复率和可判别性，作为选择正负样本的标准
    point_disc = point_disc.cpu().numpy()
    point_info = point_info.cpu().numpy()
    point_repeat = point_repeat / input_each_num
    point_disc_repeat = repeat_weight * point_repeat + (1 - repeat_weight) * point_disc + \
                        info_weight * point_info
    # 利用重判对点排序
    dr_sort_ind = np.argsort(point_disc_repeat)[::-1]
    point_dr_sort = point_disc_repeat[dr_sort_ind]
    sort_y_ref = repeat_y_ref[dr_sort_ind]
    sort_x_ref = repeat_x_ref[dr_sort_ind]

    # 根据点数目分布区间得到正样本数目
    fore_num = -1
    if point_dr_sort.size >= min_point_num:
        fore_num = np.sum(point_dr_sort >= point_dr_sort[min_point_num - 1])
        fore_num = int(fore_num)
        if not (min_point_num <= fore_num <= max_point_num):
            fore_num = -1

    # 需换回
    if (point_dr_sort.size < min_point_num) or (fore_num <= 0):
        print_dict = get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                                    back_loss_cum, repeat_mean_cum, image_base_num,
                                    fore_num_cum_runtime, image_real_num,
                                    disc_ratio, disc_value, info_value)
        return 0, False, 0, print_dict

    preserve_num = 0
    if point_dr_sort.size > max_point_num:
        preserve_num = np.sum(point_dr_sort >= point_dr_sort[max_point_num - 1])
        preserve_num = int(preserve_num)

    # 考虑到可能存在无效点（极少出现，点过多出现在无效区域），重新计算参与优化描述子的点数目
    # 因为已经确保无效点具有很小的重判值，排序后无效点一定排在后部
    min_desc_num = min(fore_num, avail_disc_num)
    max_desc_num = min(max_point_num, avail_disc_num)
    disc_thre_high = np.min(point_disc[dr_sort_ind[0:min_desc_num]])
    disc_thre_low = np.min(point_disc[dr_sort_ind[0:max_desc_num]])

    # 计算高概率特征点之间的可分性，并形成可分性损失
    loss_disc = 0
    loss_disc_exist = False
    disc_fore_num = min(candidate_num - out_num, max_desc_num)
    if disc_fore_num > 1:
        disc_y_ref = sort_y_ref[:disc_fore_num]
        disc_x_ref = sort_x_ref[:disc_fore_num]
        # 计算可分性损失
        pair_thre_fore = {'need_posi_num': 0.5, 'need_neg_num_in': 0.5,
                          'need_neg_num_out': 0.5}
        disc_corr_weight = max(input_each_num / disc_fore_num, disc_corr_weight_min)
        fore_disc = calcu_desc_dist(desc, H, range_now, disc_x_ref, disc_y_ref,
                                    xy_range, dist_value_config, pair_thre_fore, disc_corr_weight, local_rad)
        fore_disc = fore_disc[fore_disc > out_disc_value]
        if fore_disc.shape[0] > 0:
            loss_disc = -torch.mean(fore_disc)
            loss_disc_exist = True
            # 可分性损失在梯度下降中不进行归一化，但显示训练进展时对其进行归一化
            disc_value = loss_disc.item()
            max_disc_value = (disc_corr_weight * dist_value_config['corr_margin'] -
                              (1 - disc_corr_weight) * dist_value_config['cross_margin'])
            min_disc_value = (disc_corr_weight * dist_value_config['cross_margin'] -
                              (1 - disc_corr_weight) * dist_value_config['corr_margin'])
            disc_ratio = (-disc_value - min_disc_value) / (max_disc_value - min_disc_value)

    # 按照点数目分布选出正样本
    fore_y_ref = sort_y_ref[:fore_num]
    fore_x_ref = sort_x_ref[:fore_num]
    fore_xy_ref = np.c_[fore_x_ref, fore_y_ref][np.newaxis, :].astype('float32')
    # 正样本邻域内的点，以及所有的保持样本点，不会再作为负样本
    preserve_y_ref = sort_y_ref[:preserve_num]
    preserve_x_ref = sort_x_ref[:preserve_num]
    fore_mask = np.zeros((image_row, image_col), dtype='bool')
    fore_mask[fore_y_ref, fore_x_ref] = True
    fore_out_mask = np.logical_not(binary_dilation(fore_mask, fore_around_win))
    fore_out_mask[preserve_y_ref, preserve_x_ref] = False
    fore_mask_inv_vec = fore_out_mask.reshape(-1)
    # back_mask = (fore_out_mask & avail_pos & (point_cum > 0))
    back1_mask = (fore_out_mask & (point_cum > 0))
    back1_y, back1_x = np.where(back1_mask)
    back1_num = min(back1_y.size, round(fore_num * back1_num_ratio))
    # 按照点数目分布选出负样本
    if 0 < back1_num < back1_y.size:
        select_ind = np.random.choice(back1_y.size, back1_num, replace=False)
        back1_y = back1_y[select_ind].astype('float32')
        back1_x = back1_x[select_ind].astype('float32')
    back1_point = np.c_[back1_x, back1_y][np.newaxis, :].astype('float32')

    # 记录特征点前景背景位置
    sample_loc = np.zeros(0, dtype='int')
    label = np.zeros(0, dtype='int')
    for image_id in range_now:
        h_now = H[image_id].numpy()
        # 随机选取back2_num_ratio比例的点作为负样本
        back2_num = round(fore_num * back2_num_ratio)
        back2_loc = np.random.randint(0, image_row * image_col, size=back2_num)
        back2_loc = back2_loc[fore_mask_inv_vec[back2_loc]]
        back2_loc = image_row * image_col * image_id + back2_loc
        back2_loc = np.unique(back2_loc)
        gt_vec[back2_loc] -= score_detach_vec[back2_loc]
        back_num_cum += back2_loc.size
        back_loss_cum += torch.sum(score_detach_vec[back2_loc]).item()
        # 设置优先的负样本
        if back1_num > 0:
            # 投影至原始图像坐标系下
            back1_point_now = cv2.perspectiveTransform(back1_point, h_now)
            back1_point_now = np.rint(back1_point_now).astype('int').squeeze(0)
            back1_point_now = remove_out_point(back1_point_now, xy_range)
            back1_loc = image_row * image_col * image_id + \
                        back1_point_now[:, 1] * image_col + back1_point_now[:, 0]
            back1_loc = np.unique(back1_loc)
            gt_vec[back1_loc] -= score_detach_vec[back1_loc]
            # gt_vec[back1_loc] = 0
            back_num_cum += back1_loc.size
            back_loss_cum += torch.sum(score_detach_vec[back1_loc]).item()

        # 设置正样本
        fore_point = cv2.perspectiveTransform(fore_xy_ref, h_now)
        fore_point = np.rint(fore_point).astype('int').squeeze(0)
        fore_point = remove_out_point(fore_point, xy_range)
        fore_loc = image_row * image_col * image_id + \
                   fore_point[:, 1] * image_col + fore_point[:, 0]
        fore_loc = np.unique(fore_loc)
        gt_vec[fore_loc] += (1 - score_detach_vec[fore_loc])
        # gt_vec[fore_loc] = 1
        fore_num_cum += fore_loc.size
        fore_loss_cum += torch.sum(1 - score_detach_vec[fore_loc]).item()

    # 对正负样本均计算重建损失，用于训练重建网络和特征点网络
    back2_recon_num = max(min_point_num, fore_num + back1_num)
    back2_x = np.random.randint(0, image_col, size=back2_recon_num)
    back2_y = np.random.randint(0, image_row, size=back2_recon_num)
    recon_x = np.r_[fore_x_ref, back1_x, back2_x]
    recon_y = np.r_[fore_y_ref, back1_y, back2_y]
    with torch.no_grad():
        _, loss_info = calcu_info(desc_detach, H, range_now,
                                  recon_x, recon_y,
                                  xy_range, out_dist, recon_target, net_recon, True)

    info_value = loss_info.item()

    print_dict = get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                                back_loss_cum, repeat_mean_cum, image_base_num,
                                fore_num_cum_runtime, image_real_num,
                                disc_ratio, disc_value, info_value)
    return loss_disc, loss_disc_exist, loss_info, print_dict
