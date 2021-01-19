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

# select interest point with a unsupervised way
# compute the loss of detection and description

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


def point_select(score_output: torch.Tensor, score_detach: torch.Tensor,
                 desc: torch.Tensor, gt: torch.Tensor, H: torch.Tensor,
                 range_now: range, local_rad: int,
                 out_rad: int, inner_rad: int, soft_dist: int,
                 back1_weight: float, out_dist: int, image_ori: torch.Tensor,
                 recon_target: torch.Tensor, net_recon, need_info_mark
                 ):
    desc_detach = desc.detach()
    gt_vec = gt.view(-1)
    score_detach_vec = score_detach.view(-1)
    MSE_loss = nn.MSELoss()
    image_row, image_col = score_detach.shape[-2:]
    score_flatten = score_detach.view(-1)
    input_each_num = len(range_now)
    # the repeatability of a point will be computed
    # if this point is visible in as least min_need_avail_num images
    min_need_avail_num = 5
    # if the number of positive samples (i.e., interest points) is less than exit_min_point_num,
    # this image will be skipped in the training
    exit_min_point_num = 3

    # the points in the local window around a interest point will not be selected as
    # training samples, and fore_around_win is used to obtain the local window
    fore_around_win = get_circle_win(local_rad, dtype='bool')
    xy_range = np.array([out_dist, image_col - out_dist, out_dist, image_row - out_dist])
    # the number of interest points should be in [min_point_num, max_point_num]
    min_point_num = 200
    max_point_num = 400
    # back1_num_ratio determines the number of hard negative samples
    # back2_num_ratio determines the number of normal negative samples
    back1_num_ratio = 0.8
    back2_num_ratio = 0.5
    # the weight of positive samples in the computation of discriminability
    disc_corr_weight_min = input_each_num / max_point_num
    # the weight of repeatability in the loss
    repeat_weight = np.random.uniform(0.5, 1)
    if need_info_mark:
        info_weight = np.random.uniform(0, 0.25)
    else:
        info_weight = 0

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

    repeat_mean_cum += repeat_num_mean
    image_base_num += 1

    # obtain all candidates of positive samples,
    # and sort them in accordance with the repeatability
    min_repeat_thre = 1
    y_total, x_total = np.where(point_cum >= min_repeat_thre)
    point_cum_total = point_cum[y_total, x_total]
    cum_total_sort_ind = np.argsort(point_cum_total)[::-1]
    cum_total_sort = point_cum_total[cum_total_sort_ind]
    y_total = y_total[cum_total_sort_ind]
    x_total = x_total[cum_total_sort_ind]
    # select the points with high repeatability as candidates
    candidate_num_init = min(1000, cum_total_sort.size)
    candidate_num_max = 2000
    candi_repeat_thre = max(cum_total_sort[candidate_num_init - 1], min_repeat_thre)
    candidate_num = np.sum(cum_total_sort >= candi_repeat_thre)

    point_repeat = cum_total_sort[:candidate_num]
    repeat_y_ref = y_total[:candidate_num]
    repeat_x_ref = x_total[:candidate_num]

    # the distance hyperparameters in the computation of discriminability
    dist_value_config = {'corr_target': 1, 'cross_target': -1,
                         'corr_margin': 1, 'cross_margin': 0.2}

    if candidate_num < exit_min_point_num:
        print_dict = get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                                    back_loss_cum, repeat_mean_cum, image_base_num,
                                    fore_num_cum_runtime, image_real_num,
                                    disc_ratio, disc_value, info_value)
        return 0, False, 0, print_dict

    pair_thre_init = {'need_posi_num': 2.5, 'need_neg_num_in': 4.5,
                      'need_neg_num_out': candidate_num / 3}
    # if the number of candidates is too large, the memory will be not enough
    # in this situation, the disc and recon will not be computed,
    # and only the repeatability will be used to select positive samples
    if candidate_num <= candidate_num_max:
        # compute the discriminability of candidate points
        point_disc = calcu_desc_dist(desc_detach, H, range_now, repeat_x_ref, repeat_y_ref,
                                     xy_range, dist_value_config, pair_thre_init, disc_corr_weight_min, local_rad)
        # max_disc_value为理论上的最大可分性取值
        max_disc_value = (disc_corr_weight_min * dist_value_config['corr_margin'] -
                          (1 - disc_corr_weight_min) * dist_value_config['cross_margin'])
        # normalize the value of discriminability
        min_disc_value = (disc_corr_weight_min * dist_value_config['cross_margin'] -
                          (1 - disc_corr_weight_min) * dist_value_config['corr_margin'])
        point_disc = (point_disc - min_disc_value) / (max_disc_value - min_disc_value)

        # if the informativeness is required, compute it without gradient
        if need_info_mark:
            point_info, _ = calcu_info(
                desc_detach, H, range_now, repeat_x_ref, repeat_y_ref,
                xy_range, out_dist, recon_target, net_recon, False)
        else:
            point_info = torch.zeros(candidate_num, dtype=desc_detach.dtype, device=desc_detach.device)
    else:
        # the number of candidates is too large,
        # directly set the disc and info as zero to save the memory,
        # which means they have no contributions to select the interest points
        point_disc = torch.zeros(candidate_num, dtype=desc_detach.dtype, device=desc_detach.device)
        point_info = torch.zeros(candidate_num, dtype=desc_detach.dtype, device=desc_detach.device)

    # some points are not in the range of images after the image transformation,
    # set their discriminability as random small values
    out_disc_value = dist_value_config['cross_target'] - dist_value_config['corr_target']
    out_disc_pos = (point_disc <= out_disc_value)
    out_num = torch.sum(out_disc_pos).item()
    point_disc[out_disc_pos] = 10 * out_disc_value - torch.rand(out_num, dtype=desc.dtype, device=desc.device)

    avail_disc_num = torch.sum(point_disc > out_disc_value).item()
    if avail_disc_num < exit_min_point_num:
        print_dict = get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                                    back_loss_cum, repeat_mean_cum, image_base_num,
                                    fore_num_cum_runtime, image_real_num,
                                    disc_ratio, disc_value, info_value)
        return 0, False, 0, print_dict

    # sort the candidates in accordance with the combination of
    # repeatability, discriminability and informativeness
    point_disc = point_disc.cpu().numpy()
    point_info = point_info.cpu().numpy()
    point_repeat = point_repeat / input_each_num
    point_disc_repeat = repeat_weight * point_repeat + (1 - repeat_weight) * point_disc + \
                        info_weight * point_info
    dr_sort_ind = np.argsort(point_disc_repeat)[::-1]
    point_dr_sort = point_disc_repeat[dr_sort_ind]
    sort_y_ref = repeat_y_ref[dr_sort_ind]
    sort_x_ref = repeat_x_ref[dr_sort_ind]

    # obtain the number of positive samples (interest points)
    fore_num = -1
    if point_dr_sort.size >= min_point_num:
        fore_num = np.sum(point_dr_sort >= point_dr_sort[min_point_num - 1])
        fore_num = int(fore_num)
        if not (min_point_num <= fore_num <= max_point_num):
            fore_num = -1

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

    min_desc_num = min(fore_num, avail_disc_num)
    max_desc_num = min(max_point_num, avail_disc_num)

    # compute the disc loss of interest point
    loss_disc = 0
    loss_disc_exist = False
    disc_fore_num = min(candidate_num - out_num, max_desc_num)
    if disc_fore_num > 1:
        disc_y_ref = sort_y_ref[:disc_fore_num]
        disc_x_ref = sort_x_ref[:disc_fore_num]
        pair_thre_fore = {'need_posi_num': 0.5, 'need_neg_num_in': 0.5,
                          'need_neg_num_out': 0.5}
        disc_corr_weight = max(input_each_num / disc_fore_num, disc_corr_weight_min)
        fore_disc = calcu_desc_dist(desc, H, range_now, disc_x_ref, disc_y_ref,
                                    xy_range, dist_value_config, pair_thre_fore, disc_corr_weight, local_rad)
        fore_disc = fore_disc[fore_disc > out_disc_value]
        if fore_disc.shape[0] > 0:
            loss_disc = -torch.mean(fore_disc)
            loss_disc_exist = True
            disc_value = loss_disc.item()
            max_disc_value = (disc_corr_weight * dist_value_config['corr_margin'] -
                              (1 - disc_corr_weight) * dist_value_config['cross_margin'])
            min_disc_value = (disc_corr_weight * dist_value_config['cross_margin'] -
                              (1 - disc_corr_weight) * dist_value_config['corr_margin'])
            # disc_ratio is the normalized value of disc loss
            disc_ratio = (-disc_value - min_disc_value) / (max_disc_value - min_disc_value)

    # obtain the coordinates of interest points
    fore_y_ref = sort_y_ref[:fore_num]
    fore_x_ref = sort_x_ref[:fore_num]
    fore_xy_ref = np.c_[fore_x_ref, fore_y_ref][np.newaxis, :].astype('float32')
    # the points near to the positive points, and the point with relatively high score,
    # will not be selected as negative points
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
    if 0 < back1_num < back1_y.size:
        select_ind = np.random.choice(back1_y.size, back1_num, replace=False)
        back1_y = back1_y[select_ind].astype('float32')
        back1_x = back1_x[select_ind].astype('float32')
    back1_point = np.c_[back1_x, back1_y][np.newaxis, :].astype('float32')

    # record the position of positive and negative samples,
    # which is convenient to compute the loss of detection
    sample_loc = np.zeros(0, dtype='int')
    label = np.zeros(0, dtype='int')
    for image_id in range_now:
        h_now = H[image_id].numpy()
        # select the normal negative samples randomly
        back2_num = round(fore_num * back2_num_ratio)
        back2_loc = np.random.randint(0, image_row * image_col, size=back2_num)
        back2_loc = back2_loc[fore_mask_inv_vec[back2_loc]]
        back2_loc = image_row * image_col * image_id + back2_loc
        back2_loc = np.unique(back2_loc)
        gt_vec[back2_loc] -= score_detach_vec[back2_loc]
        back_num_cum += back2_loc.size
        back_loss_cum += torch.sum(score_detach_vec[back2_loc]).item()
        # select the hard negative samples
        if back1_num > 0:
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

        # record the positive samples
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

    # compute the reconstruction loss for both the positive and negative samples
    # note the recon loss is only used to shown the training status but not used to computed the gradient
    # as introduced in the paper, informativeness is used to select the interest point
    back2_recon_num = max(min_point_num, fore_num + back1_num)
    back2_x = np.random.randint(0, image_col, size=back2_recon_num)
    back2_y = np.random.randint(0, image_row, size=back2_recon_num)
    recon_x = np.r_[fore_x_ref, back1_x, back2_x]
    recon_y = np.r_[fore_y_ref, back1_y, back2_y]
    if need_info_mark:
        with torch.no_grad():
            _, loss_info = calcu_info(desc_detach, H, range_now,
                                      recon_x, recon_y,
                                      xy_range, out_dist, recon_target, net_recon, True)
    else:
        loss_info = torch.zeros(1)

    info_value = loss_info.item()

    print_dict = get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                                back_loss_cum, repeat_mean_cum, image_base_num,
                                fore_num_cum_runtime, image_real_num,
                                disc_ratio, disc_value, info_value)
    return loss_disc, loss_disc_exist, loss_info, print_dict
