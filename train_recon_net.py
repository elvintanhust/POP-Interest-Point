import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
import warnings
import glob
import os
from torch.utils.data import DataLoader
from reconstruct_net import ReconstructNet
from dataset_class import PointDataset
from point_select_utils import get_multi_patch

warnings.filterwarnings("ignore")


# train the reconstructor

def train_recon(args):
    write_dir = args.save_path
    device = torch.device(args.device)
    save_epoch_interval=args.save_epoch_interval
    print_interval = args.print_iter_interval
    save_interval = args.save_iter_interval

    point_dataset = PointDataset(args.train_image_path, args.random_aug_num,
                                 args.train_height, args.train_width)
    dataloader = DataLoader(point_dataset, batch_size=1, shuffle=True,
                            num_workers=args.data_num_workers)

    net_recon = ReconstructNet(feature_len=64, in_f_len=64)
    epoch_init = 0
    need_checkpoint = False
    if args.recon_checkpoint_path is not None:
        need_checkpoint = True
        checkpoint = torch.load(args.recon_checkpoint_path, map_location=device)
        net_recon.load_state_dict(checkpoint['model_state_dict_recon'], strict=True)
        net_recon.train()
        epoch_init = checkpoint['epoch']
    net_recon.to(device)

    optimizer_recon = optim.Adam(net_recon.parameters())
    if need_checkpoint:
        optimizer_recon.load_state_dict(checkpoint['optimizer_recon_state_dict'])

    loss_MSE = nn.MSELoss()

    image_row, image_col = point_dataset.image_row, point_dataset.image_col
    # the size of local patch
    patch_rad = 8
    patch_size = 2 * patch_rad
    # the number of patches
    patch_num = 300

    cum_iter_num = 0
    max_epoch = epoch_init + args.max_epoch
    batch_num = len(dataloader)
    for epoch in range(epoch_init, max_epoch):
        for idx, batch in enumerate(dataloader, 0):
            cum_iter_num += 1
            optimizer_recon.zero_grad()
            input = batch['image']
            input = input.view(tuple(np.r_[input.shape[0] * input.shape[1],
                                           input.shape[2:]]))
            input = input.to(device)

            gt_patch = torch.zeros((input.shape[0] * patch_num, 3,
                                    patch_size, patch_size), device=device)
            for image_id in range(input.shape[0]):
                r_start = np.random.randint(0, image_row - 2 * patch_rad - 1, size=(patch_num))
                c_start = np.random.randint(0, image_col - 2 * patch_rad - 1, size=(patch_num))
                r_len = patch_size
                c_len = patch_size
                gt_patch[image_id * patch_num:(image_id + 1) * patch_num] = \
                    get_multi_patch(input[image_id], r_start, c_start, r_len, c_len)

            patch_recon = net_recon(gt_patch)

            gt_temp = (gt_patch.cpu().numpy().transpose((0, 2, 3, 1)) * 129 + 128).astype('uint8')
            gt_temp = np.minimum(np.maximum(gt_temp, 0), 255).astype('uint8')
            recon_temp = (patch_recon.detach().cpu().numpy().transpose((0, 2, 3, 1)) * 129 + 128).astype('uint8')
            recon_temp = np.minimum(np.maximum(recon_temp, 0), 255).astype('uint8')

            loss = loss_MSE(patch_recon, gt_patch)

            loss.backward()
            optimizer_recon.step()

            if cum_iter_num % print_interval == print_interval - 1:
                print('ep:%d' % epoch, 'iter:%d/%d' % (idx, batch_num - 1),
                      ' loss:%.4f' % loss.item())

            if cum_iter_num % save_interval == save_interval - 1:
                torch.save({
                    'model_state_dict_recon': net_recon.state_dict(),
                    'optimizer_recon_state_dict': optimizer_recon.state_dict(),
                    'epoch': epoch,
                }, os.path.join(write_dir, 'recon_net.pth'))

        if epoch % save_epoch_interval == save_epoch_interval - 1:
            torch.save({
                'model_state_dict_recon': net_recon.state_dict(),
                'optimizer_recon_state_dict': optimizer_recon.state_dict(),
                'epoch': epoch,
            }, os.path.join(write_dir, 'recon_net_epoch_end%d.pth' % epoch))


def main():
    parser = argparse.ArgumentParser(description="Reconstructor Training")

    parser.add_argument('--train-image-path', type=str,
                        default='demo_input_images',
                        help='the path of training images, which should be ' \
                             'a directory containing only images')
    parser.add_argument('--recon-checkpoint-path', type=str,
                        default=None,
                        help='set the path of checkpoint if needed, ' \
                             'and use the default when train the model from scratch')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='the device used to train the model, ' \
                             'which is represented with the pytorch format')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='the maximal number of epochs')
    parser.add_argument('--data-num-workers', type=int, default=2,
                        help='the num_workers of DataLoader')
    parser.add_argument('--train-width', type=int, default=320,
                        help='every training image will be resized with train_width')
    parser.add_argument('--train-height', type=int, default=240,
                        help='every training image will be resized with train_height')
    parser.add_argument('--save-epoch-interval', type=int, default=1,
                        help='the model will be saved every ' \
                             'save_epoch_interval iterations')
    parser.add_argument('--save-path', type=str,
                        default='save_recon_model',
                        help='the path used to save the trained model')
    parser.add_argument('--print-iter-interval', type=int, default=4,
                        help='the training status will be printed every ' \
                             'print_iter_interval iterations')
    parser.add_argument('--save-iter-interval', type=int, default=1000,
                        help='the temporary model will be saved every ' \
                             'save_iter_interval iteration, and the filename is recon_net_temp.pth')
    parser.add_argument('--random-aug-num', type=int, default=10,
                        help='every training image will be transformed to ' \
                             'random_aug_num new images')

    args = parser.parse_args()
    # create the storage directory if needed
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    train_recon(args)


if __name__ == '__main__':
    main()
