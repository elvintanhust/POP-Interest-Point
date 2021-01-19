import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import glob
import imgaug as ia
import math
from torch.utils.data import Dataset
from imgaug import augmenters as iaa

ia.seed(random.randint(0, 100000))
avail_image_ext = ['.jpg', '.jpeg', '.tif', '.tiff', '.png',
                   '.bmp', '.gif', '.ppm']

# several dataset class

def filter_out_non_image(image_names_all, print_mark=False):
    # check the ext
    ext = [os.path.splitext(name)[1] for name in image_names_all]
    all_num = len(image_names_all)
    image_names = [image_names_all[pos] for pos in range(all_num) if
                   ext[pos] in avail_image_ext]
    ignore_num = len(image_names_all) - len(image_names)
    if print_mark:
        if ignore_num == 1:
            print('one file is ignored because its '
                  'extension is not in %s' % (', '.join(avail_image_ext)))
        elif ignore_num > 1:
            print('%d files are ignored because their '
                  'extensions are not in %s' % (ignore_num, ', '.join(avail_image_ext)))

    return image_names


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose([0, 3, 1, 2])
        base_gray = sample['base_gray'][np.newaxis, :]
        # here normalize landmarks to [0,1]
        sample_tensor = {
            'image': (torch.from_numpy(image).float() - 128) / 129,
            # 'image': (torch.from_numpy(image).float() / 256),
            'H': sample['H'],
            'image_ori': sample['image_ori'],
            'view_batch_mark': sample['view_batch_mark'],
            'base_gray': (torch.from_numpy(base_gray).float() - 128) / 129}
        return sample_tensor


class PointDataset(Dataset):
    def __init__(self, root_dir: str,
                 random_num: int, image_row: int, image_col: int):
        self.random_num = random_num
        self.root_dir = root_dir
        image_names_all = glob.glob(os.path.join(self.root_dir, '*.*'))
        # check the ext
        self.image_names = filter_out_non_image(image_names_all, print_mark=True)

        self.image_row = image_row
        self.image_col = image_col
        self.to_tensor_fun = ToTensor()
        self.ill_seq = self.get_ill_seq()
        self.simple_ill_seq = self.get_simple_ill_seq()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_name = self.image_names[item]

        image_ori_base = cv2.imread(image_name)
        image_ori_base = cv2.cvtColor(image_ori_base, cv2.COLOR_BGR2RGB)

        # 随机翻转
        image_ori_base = self.flip_transpose_image(image_ori_base)
        # 计算缩放到固定尺寸所需的单应矩阵
        row_ori, col_ori = image_ori_base.shape[:2]
        x_ratio = self.image_col / col_ori
        y_ratio = self.image_row / row_ori
        resize_homo = np.array([[x_ratio, 0, 0], [0, y_ratio, 0], [0, 0, 1]],
                               dtype='float32')

        row, col = self.image_row, self.image_col
        range_ratio = 1 / 4
        range_row, range_col = round(row * range_ratio), round(col * range_ratio)
        random_num = self.random_num
        # 以shrink_prob的概率对图像进行缩小型变换（而不是普通的单应变换）
        shrink_prob = 0.3
        shr_in_ratio = 1 / 3
        shr_out_ratio = 1 / 15
        shr_in_row, shr_in_col = round(row * shr_in_ratio), round(col * shr_in_ratio)
        shr_out_row, shr_out_col = round(row * shr_out_ratio), round(col * shr_out_ratio)
        # 以extend_prob的概率对图像进行放大型变换
        extend_prob = 0.3
        ext_in_ratio = 1 / 15
        ext_out_ratio = 1 / 2
        ext_in_row, ext_in_col = round(row * ext_in_ratio), round(col * ext_in_ratio)
        ext_out_row, ext_out_col = round(row * ext_out_ratio), round(col * ext_out_ratio)
        # 以rotate_prob的概率进行显著旋转，独立于尺度操作
        rotate_prob = 0.3
        # rotate_prob = -1
        rotate_range = [math.pi / 6, math.pi * 7 / 18]
        # rotate_range = [math.pi / 4, math.pi * 7 / 4]
        row_half = round(row / 2)
        col_half = round(col / 2)
        d_lt_h = np.array([[1, 0, -col_half], [0, 1, -row_half],
                           [0, 0, 1]], dtype='float32')
        d_rb_h = np.array([[1, 0, col_half], [0, 1, row_half],
                           [0, 0, 1]], dtype='float32')
        center_rot_h = np.eye(3)

        image_ori = np.tile(image_ori_base, [random_num, 1, 1, 1])
        # 记录当前批次用于训练视角任务
        view_prob = -1
        # view_prob = 2
        view_batch_mark = False
        if random.random() < view_prob:
            image_ori = self.random_ill_change(image_ori, is_simple=True)
            view_batch_mark = True
        else:
            image_ori = self.random_ill_change(image_ori, is_simple=False)

        # 用来记录变换后的图像
        image_new = np.zeros([random_num, row, col, 3], dtype=image_ori_base.dtype)
        H = np.zeros((random_num, 3, 3), dtype='float32')
        for image_id in range(random_num):
            point_src = np.array([[1, 1], [col, 1], [1, row], [col, row]], dtype='float32')
            while True:
                # 以一定概率进行随机的显著旋转,仅在视角任务中使用
                rotate_mark = False
                random_value = random.random()
                if view_batch_mark and random_value < rotate_prob:
                    rotate_mark = True
                    angle = np.random.uniform(*rotate_range)
                    # 随机确定旋转方向
                    if random_value < rotate_prob / 2:
                        angle = -angle
                    sin_angle = math.sin(angle)
                    cos_angle = math.cos(angle)
                    rotate_h = np.array([[cos_angle, -sin_angle, 0],
                                         [sin_angle, cos_angle, 0],
                                         [0, 0, 1]], dtype='float32')
                    center_rot_h = np.dot(d_rb_h, np.dot(rotate_h, d_lt_h))

                random_value = random.random()
                point_dst = point_src.copy()
                if random_value < shrink_prob:
                    point_dst[np.array([0, 2]), np.array([0, 0])] += \
                        np.random.randint(-shr_out_col, shr_in_col, 2)
                    point_dst[np.array([0, 1]), np.array([1, 1])] += \
                        np.random.randint(-shr_out_row, shr_in_row, 2)
                    point_dst[np.array([1, 3]), np.array([0, 0])] += \
                        np.random.randint(-shr_in_col, shr_out_col, 2)
                    point_dst[np.array([2, 3]), np.array([1, 1])] += \
                        np.random.randint(-shr_in_row, shr_out_row, 2)
                elif random_value < shrink_prob + extend_prob:
                    point_dst[np.array([0, 2]), np.array([0, 0])] += \
                        np.random.randint(-ext_out_col, ext_in_col, 2)
                    point_dst[np.array([0, 1]), np.array([1, 1])] += \
                        np.random.randint(-ext_out_row, ext_in_row, 2)
                    point_dst[np.array([1, 3]), np.array([0, 0])] += \
                        np.random.randint(-ext_in_col, ext_out_col, 2)
                    point_dst[np.array([2, 3]), np.array([1, 1])] += \
                        np.random.randint(-ext_in_row, ext_out_row, 2)
                else:
                    point_dst += np.c_[np.random.randint(-range_col, range_col, 4),
                                       np.random.randint(-range_row, range_row, 4)].astype('float32')

                h_now = cv2.getPerspectiveTransform(point_src, point_dst)
                # 加入随机旋转
                if rotate_mark:
                    h_now = np.dot(h_now, center_rot_h)

                # 确保得到的单应矩阵是可逆的（原则上不会出现不满足的情况）
                if abs(np.linalg.det(h_now)) > 1e-5:
                    break
            # 与尺寸缩放矩阵相乘得到最终的变换矩阵
            h_ori_now = np.dot(h_now, resize_homo)
            image_new[image_id] = cv2.warpPerspective(image_ori[image_id], h_ori_now, (col, row))
            # 切记此处需要记录的变换矩阵H不包含resize变换
            # 因为H的目的在于将坐标缩放至固定尺寸(self.image_row,self.image_col)下
            H[image_id] = h_now

        image = image_new.copy().astype('float32')

        base_resize = cv2.warpPerspective(image_ori_base, resize_homo, (col, row))
        base_gray = cv2.cvtColor(base_resize, cv2.COLOR_RGB2GRAY)
        sample = {'image': image, 'H': H,
                  'image_ori': image_new, 'view_batch_mark': view_batch_mark,
                  'base_gray': base_gray}

        sample = self.to_tensor_fun(sample)

        return sample

    def flip_transpose_image(self, image_ori):
        row, col = image_ori.shape[:2]
        rand_value = random.random()
        if rand_value < 0.6:
            rand_code = 1
        elif rand_value < 0.8:
            rand_code = 0
        else:
            rand_code = -1
        image_new = cv2.flip(image_ori, flipCode=rand_code)
        if len(image_new.shape) < 3:
            image_new = image_new[:, np.newaxis]
        # 图像原始尺寸与预期的长宽大小不一致，则旋转90度
        if (row < col) ^ (self.image_row < self.image_col):
            image_new = image_new.transpose((1, 0, 2))
        return image_new

    def resize_image(self, image_ori):
        row, col = image_ori.shape[:2]
        image_new = image_ori
        if (row != self.image_row) or (col != self.image_col):
            image_new = cv2.resize(image_ori, (self.image_col, self.image_row))
        return image_new

    def random_ill_change(self, image_ori, is_simple=False):
        if is_simple:
            image_new = self.simple_ill_seq.augment_images(image_ori)
        else:
            image_new = self.ill_seq.augment_images(image_ori)

        image_num = image_new.shape[0]
        shadow_num = round(image_num / 3)
        for image_id in tuple(np.random.choice(np.arange(image_num),
                                               shadow_num, replace=False)):
            # continue
            image_new[image_id] = self.insert_shadow(image_new[image_id], is_simple)
        return image_new

    def insert_shadow(self, image_ori, is_simple=False):
        img_new = image_ori.copy()
        image_row, image_col = image_ori.shape[:2]
        shadow_num = random.randint(5, 50)
        min_size, max_size = 10, round(min(image_ori.shape[:2]) / 4)
        mask = np.zeros(image_ori.shape[:2], np.uint8)
        rect_shrink_ratio = 1 / 3
        ellipse_prob = 0.3
        transparency_range = [0.2, 0.6]
        if is_simple:
            transparency_range = [0.6, 0.9]

        for i in range(shadow_num):
            # 阴影尺寸
            ax = random.randint(min_size, max_size)
            ay = random.randint(min_size, max_size)
            max_rad = max(ax, ay)
            # 阴影中心
            x = np.random.randint(max_rad, image_col - max_rad)
            y = np.random.randint(max_rad, image_row - max_rad)
            # 选取阴影形状
            if random.random() < ellipse_prob:
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)
            else:
                shr_x_range = round(rect_shrink_ratio * ax)
                shr_y_range = round(rect_shrink_ratio * ay)
                rad_x, rad_y = round(ax / 2), round(ay / 2)
                rect_point = np.array([[x - rad_x, y - rad_y], [x + rad_x, y - rad_y],
                                       [x + rad_x, y + rad_y], [x - rad_x, y + rad_y]], dtype='int32')
                rect_point += np.c_[np.random.randint(-shr_x_range, shr_x_range, 4),
                                    np.random.randint(-shr_y_range, shr_y_range, 4)]
                cv2.fillConvexPoly(mask, rect_point, 255)

        mask = mask > 1
        mask = np.tile(np.expand_dims(mask, axis=2), (1, 1, 3))
        transparency = np.random.uniform(*transparency_range)
        shadow_value = random.choice((0, 255))
        img_new[mask] = img_new[mask] * transparency + shadow_value * (1 - transparency)

        return img_new

    def get_ill_seq(self):
        light_change = 50
        seq = iaa.Sequential([
            # 全局调整，含有颜色空间调整
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=iaa.OneOf([
                        iaa.WithChannels(0, iaa.Add((-5, 5))),
                        iaa.WithChannels(1, iaa.Add((-20, 20))),
                        iaa.WithChannels(2, iaa.Add((-light_change, light_change))),
                    ])
                ),
                iaa.Grayscale((0.2, 0.6)),
                iaa.ChannelShuffle(1),
                iaa.Add((-light_change, light_change)),
                iaa.Multiply((0.5, 1.5)),
            ])),

            # # dropout阴影模仿，暂时不使用，转而使用了自定义的阴影模仿
            # iaa.Sometimes(0.5, iaa.OneOf([
            #     iaa.Alpha((0.2, 0.7), iaa.CoarseDropout(p=0.2, size_percent=(0.02, 0.005)))
            # ])),

            # 椒盐噪声
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.Alpha((0.2, 0.6), iaa.SaltAndPepper((0.01, 0.03)))
            ])),

            # 图像反转
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.Invert(1),
            ])),

            # 对比度调整
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.ContrastNormalization((0.5, 1.5)),
            ])),

            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.AdditiveGaussianNoise(0, (3, 6)),
                iaa.AdditivePoissonNoise((3, 6)),
                iaa.JpegCompression((30, 60)),
                iaa.GaussianBlur(sigma=1),
                iaa.AverageBlur((1, 3)),
                iaa.MedianBlur((1, 3)),
            ])),
        ])
        return seq

    def get_simple_ill_seq(self):
        light_change = 20
        seq = iaa.Sequential([
            # 全局调整，含有颜色空间调整
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=iaa.OneOf([
                        iaa.WithChannels(0, iaa.Add((-5, 5))),
                        iaa.WithChannels(1, iaa.Add((-20, 20))),
                        iaa.WithChannels(2, iaa.Add((-light_change, light_change))),
                    ])
                ),
                iaa.Grayscale((0.2, 0.6)),
                iaa.Add((-light_change, light_change)),
                iaa.Multiply((0.8, 1.2)),
            ])),

            # 椒盐噪声
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.Alpha((0.2, 0.6), iaa.SaltAndPepper((0.01, 0.03)))
            ])),

            # 对比度调整
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.ContrastNormalization((0.8, 1.2)),
            ])),

            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.AdditiveGaussianNoise(0, 1),
                iaa.AdditivePoissonNoise(1),
                iaa.JpegCompression((30, 60)),
                iaa.GaussianBlur(sigma=1),
                iaa.AverageBlur(1),
                iaa.MedianBlur(1),
            ])),
        ])
        return seq


class ToTensorHPatches(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose([0, 3, 1, 2])
        image_gray = sample['image_gray']
        # here normalize landmarks to [0,1]
        sample_tensor = {'image': (torch.from_numpy(image).float() - 128) / 129,
                         'H': sample['H'],
                         'image_shape': sample['image_shape'],
                         'image_name': sample['image_name'],
                         'image_gray': torch.from_numpy(image_gray).float() / 255,
                         'image_ori': sample['image_ori']}
        return sample_tensor


class HPatchesDataset(Dataset):
    def __init__(self, root_dir: str,
                 image_row: int, image_col: int, type_str: str):
        self.root_dir = root_dir
        self.image_row = image_row
        self.image_col = image_col
        self.type_str = type_str

        if self.type_str == 'v':
            self.dir_names = glob.glob(os.path.join(self.root_dir, 'v_*'))
        elif self.type_str == 'i':
            self.dir_names = glob.glob(os.path.join(self.root_dir, 'i_*'))
        else:
            self.dir_names = glob.glob(os.path.join(self.root_dir, '*'))

        self.to_tensor_fun = ToTensorHPatches()

    def __len__(self):
        return len(self.dir_names)

    def __getitem__(self, item):
        # HPatchesDataset中取出的彩色图像数据均为RGB顺序
        dir_name = self.dir_names[item]
        image_names_all = glob.glob(os.path.join(dir_name, '*.*'))
        # check the ext
        image_names = filter_out_non_image(image_names_all, print_mark=False)
        image_names = sorted(image_names)
        image_num = len(image_names)
        assert image_num > 1

        ext = os.path.splitext(image_names[0])[1]
        image_names = glob.glob(os.path.join(dir_name, '*' + ext))
        image_num = len(image_names)

        # 显存原因，限制最大数目
        image_num = min(20, image_num)

        # 获取和设置基准图像（第一个图像）的数据和信息
        base_number = 1
        image_name_now = os.path.join(dir_name, '%d%s' % (base_number, ext))
        image_base = cv2.imread(image_name_now)
        image_base = cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB)
        image_base_shape = np.array(image_base.shape)
        H_base = np.eye(3, dtype='float32')
        image_base = cv2.resize(image_base, (self.image_col, self.image_row))
        image_base_gray = cv2.cvtColor(image_base, cv2.COLOR_RGB2GRAY)

        # 记录每个图像的数据和信息，使用第一个图像的数据进行初始化
        image_array = np.tile(image_base, [image_num, 1, 1, 1])
        H = np.tile(H_base, [image_num, 1, 1])
        image_shape = np.tile(image_base_shape, [image_num, 1])
        image_gray_array = np.tile(image_base_gray, [image_num, 1, 1, 1])
        image_ori = np.tile(image_base, [image_num, 1, 1, 1])

        for image_id in range(1, image_num):
            image_now = cv2.imread(os.path.join(dir_name, '%d%s' % (image_id + 1, ext)))
            image_now = cv2.cvtColor(image_now, cv2.COLOR_BGR2RGB)
            image_shape[image_id] = np.array(image_now.shape)
            image_array[image_id] = cv2.resize(image_now, (self.image_col, self.image_row))
            H_name_now = 'H_%d_%d' % (base_number, image_id + 1)
            H_fullname_now = os.path.join(dir_name, H_name_now)
            if os.path.exists(H_fullname_now):
                H[image_id] = np.loadtxt(H_fullname_now)
            else:
                # 默认为恒等变换，对于webcam数据集而言直接可用
                H[image_id] = np.eye(3, dtype='float32')
            image_gray_array[image_id][0] = cv2.cvtColor(image_array[image_id], cv2.COLOR_RGB2GRAY)
            image_ori[image_id] = image_array[image_id].copy()

            # # 验证H读取正确
            # image_new1 = cv2.warpPerspective(
            #     image_now, np.linalg.inv(H[image_id]),
            #     (image_base_shape[1], image_base_shape[0]))
            # image1 = cv2.resize(image_base,
            #                     (image_base_shape[1], image_base_shape[0]))
            # plt.figure(1)
            # plt.imshow(image1)
            # plt.figure(2)
            # plt.imshow(image_new1)
            # plt.show()

        sample = {'image': image_array, 'H': H, 'image_shape': image_shape,
                  'image_name': os.path.basename(dir_name),
                  'image_gray': image_gray_array,
                  'image_ori': image_ori}

        sample = self.to_tensor_fun(sample)

        return sample


class ImageSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_set_path, image_rc=None):
        self.image_size = image_rc
        self.image_row = -1
        self.image_col = -1
        if image_rc is not None:
            self.image_size = (image_rc[1], image_rc[0])
            self.image_row = image_rc[0]
            self.image_col = image_rc[1]

        names = os.listdir(image_set_path)
        # check the ext
        names = filter_out_non_image(names, print_mark=True)

        names = sorted(names)
        self.names = [os.path.join(image_set_path, name_now) for name_now in names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:
            image = cv2.resize(image, self.image_size)
        row, col = image.shape[0], image.shape[1]
        tensor1 = self.to_tensor(image)
        sample = {'image': tensor1, 'image_ori': image,
                  'image_name': os.path.basename(name)}
        return sample

    def to_tensor(self, image_np):
        tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float()
        tensor = (tensor - 127) / 128
        return tensor
