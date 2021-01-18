import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from reconstruct_net import ReconstructNet
from dataset_class import ImageSet
from torch.utils.data import DataLoader
from point_select_utils import get_multi_patch


def visualize_recon(args):
    result_dir = args.result_save_path

    # 可视化特征点检测结果
    device = torch.device(args.device)
    checkpoint = torch.load(args.recon_model_path, map_location=device)

    # 加载重建网络
    net_recon = ReconstructNet(feature_len=64, in_f_len=64)
    net_recon.load_state_dict(checkpoint['model_state_dict_recon'], strict=True)
    net_recon.eval()
    net_recon.to(device)

    patch_size = 16
    stride = 16
    each_recon_num = 1024
    score_weight = 0.5

    image_channels = 3
    image_set = ImageSet(args.input_image_path)
    dataloader = DataLoader(image_set, batch_size=1, shuffle=False)
    image_num = len(dataloader)
    for idx, batch in enumerate(dataloader, 0):
        input, image_ori, image_name = batch['image'], batch['image_ori'], \
                                       batch['image_name']

        batch_size = input.size(0)
        assert batch_size == 1
        image_ori = image_ori.numpy()
        image_name = image_name[0]
        # 用于决定写出的图像后缀
        image_base = os.path.splitext(image_name)[0]
        image_ext = os.path.splitext(image_name)[1]

        input = input.to(device)
        channels, image_row, image_col = input.shape[1:]

        r_vec_single = np.arange(0, image_row - patch_size + 1, stride)
        c_vec_single = np.arange(0, image_col - patch_size + 1, stride)
        r_mat = np.tile(r_vec_single[:, np.newaxis], (1, c_vec_single.shape[0]))
        c_mat = np.tile(c_vec_single[np.newaxis, :], (r_vec_single.shape[0], 1))
        r_vec = r_mat.reshape(-1)
        c_vec = c_mat.reshape(-1)

        patch_num = r_vec.shape[0]
        split_num = math.ceil(patch_num / each_recon_num)
        input_need = torch.zeros(channels, r_vec[-1] + patch_size, c_vec[-1] + patch_size,
                                 device=device)
        recon_full = torch.zeros(channels, r_vec[-1] + patch_size, c_vec[-1] + patch_size,
                                 device=device)
        info_mat = torch.zeros(r_vec_single.size, c_vec_single.size, device=device)
        for split_id in range(split_num):
            pos1 = split_id * each_recon_num
            pos2 = min((split_id + 1) * each_recon_num, patch_num)
            r_start = r_vec[pos1:pos2]
            c_start = c_vec[pos1:pos2]
            input_patches = get_multi_patch(input, r_start, c_start, patch_size, patch_size)
            with torch.no_grad():
                recon_patch = net_recon(input_patches)
            diff = (recon_patch - input_patches) * (recon_patch - input_patches)
            info_now = torch.mean(torch.mean(torch.mean(diff, dim=3), dim=2), dim=1)

            for p_id in range(r_start.size):
                recon_full[:, r_start[p_id]:r_start[p_id] + patch_size,
                c_start[p_id]:c_start[p_id] + patch_size] = recon_patch[p_id]
                input_need[:, r_start[p_id]:r_start[p_id] + patch_size,
                c_start[p_id]:c_start[p_id] + patch_size] = input_patches[p_id]
            info_mat[np.rint(r_start / patch_size), np.rint(c_start / patch_size)] = info_now

        recon_image = np.minimum(np.maximum((recon_full.squeeze(0) * 128 + 127).cpu().numpy(),
                                            0), 255).astype('uint8').transpose((1, 2, 0))
        image_need = np.minimum(np.maximum((input_need.squeeze(0) * 128 + 127).cpu().numpy(),
                                           0), 255).astype('uint8').transpose((1, 2, 0))
        info_image = info_mat.cpu().numpy()
        info_image = np.minimum(info_image / (np.max(info_image) / 1.3), 1)
        info_image = (info_image * 255).astype('uint8')
        info_image = cv2.resize(info_image, (image_need.shape[1], image_need.shape[0]))
        info_color = cv2.applyColorMap(255 - info_image, cv2.COLORMAP_JET)

        image_need = cv2.cvtColor(image_need, cv2.COLOR_RGB2BGR)
        recon_image = cv2.cvtColor(recon_image, cv2.COLOR_RGB2BGR)

        score_color = np.concatenate(recon_image)
        score_image = np.rint(score_weight * info_color + (1 - score_weight) * image_need).astype('uint8')
        cv2.imwrite('%s/%s%s' % (result_dir, image_base, image_ext), image_need)
        cv2.imwrite('%s/%s_recon%s' % (result_dir, image_base, image_ext), recon_image)
        cv2.imwrite('%s/%s_score%s' % (result_dir, image_base, image_ext),
                    cv2.cvtColor(score_image, cv2.COLOR_RGB2BGR))

        print('%d/%d: the image %s is reconstructed' %
              (idx + 1, image_num, image_name))


def main():
    parser = argparse.ArgumentParser(description="Perform reconstruction")

    parser.add_argument('--input-image-path', type=str,
                        default='demo_input_images',
                        help='the path of input images, which should be ' \
                             'a directory containing only images')
    parser.add_argument('--recon-model-path', type=str,
                        default='save_recon_model/recon_net_pretrained.pth',
                        help='set the path of the reconstruction network')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='the device used to train the model, ' \
                             'which is represented with the pytorch format')
    parser.add_argument('--result-save-path', type=str,
                        default='recon_image_results',
                        help='the path to save the reconstruction results')

    args = parser.parse_args()
    # create the storage directory if needed
    if not os.path.exists(args.result_save_path):
        os.mkdir(args.result_save_path)

    visualize_recon(args)


if __name__ == '__main__':
    main()
