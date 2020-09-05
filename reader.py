# import os
import sys
import cv2
# import math
import random
import functools

try:
    import cPickle as pickle
    # from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import paddle
from PIL import Image
import logging

logger = logging.getLogger(__name__)
python_ver = sys.version_info


### 读取 Kinetics 数据集的类
class KineticsReader(object):
    """
    Data reader for kinetics dataset of two format mp4 and pkl.
    1. mp4, the original format of kinetics400
    2. pkl, the mp4 was decoded previously and stored as pkl
    In both case, load the data, and then get the frame data in the form of numpy and label as an integer.
     dataset cfg: format
                  num_classes
                  seg_num
                  short_size
                  target_size
                  num_reader_threads
                  buf_size
                  image_mean
                  image_std
                  batch_size
                  list
    """

    def __init__(self, name, mode, cfg):
        self.cfg = cfg
        self.mode = mode
        self.name = name
        self.format = cfg.MODEL.format

        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.seg_num = self.get_config_from_sec('model', 'seg_num')
        self.seglen = self.get_config_from_sec('model', 'seglen')

        self.seg_num = self.get_config_from_sec(mode, 'seg_num', self.seg_num)
        self.short_size = self.get_config_from_sec(mode, 'short_size')
        self.target_size = self.get_config_from_sec(mode, 'target_size')
        self.num_reader_threads = self.get_config_from_sec(mode, 'num_reader_threads')
        self.buf_size = self.get_config_from_sec(mode, 'buf_size')
        self.enable_ce = self.get_config_from_sec(mode, 'enable_ce')

        self.img_mean = np.array(cfg.MODEL.image_mean).reshape([3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape([3, 1, 1]).astype(np.float32)
        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']
        if self.enable_ce:
            random.seed(0)
            np.random.seed(0)


    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)

    ### 在train.py 中，调用了下面 create_reader() 的方法
    def create_reader(self):
        _reader = self._reader_creator(self.filelist, 
                                    self.mode, 
                                    seg_num=self.seg_num, 
                                    seglen=self.seglen,
                                    short_size=self.short_size,
                                    target_size=self.target_size,
                                    img_mean=self.img_mean,
                                    img_std=self.img_std,
                                    shuffle=(self.mode == 'train'),
                                    num_threads=self.num_reader_threads,
                                    buf_size=self.buf_size,          
                                    format=self.format)

        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return _batch_reader

    def _reader_creator(self,  pickle_list, mode, seg_num, seglen,
                        short_size, target_size, img_mean, img_std,
                        shuffle=False, num_threads=1, buf_size=1024, format='pkl'):
        def decode_mp4(sample, mode, seg_num, seglen, short_size, target_size, img_mean, img_std):
            sample = sample[0].split(' ')
            mp4_path = sample[0]
            # when infer, we store vid as label
            label = int(sample[1])
            try:
                imgs = mp4_loader(mp4_path, seg_num, seglen, mode)
                if len(imgs) < 1:
                    logger.error('{} frame length {} less than 1.'.format( mp4_path, len(imgs)))
                    return None, None
            except:
                logger.error('Error when loading {}'.format(mp4_path))
                return None, None

            return imgs_transform(imgs, label, mode, seg_num, seglen, short_size, target_size, img_mean, img_std)

        def decode_pickle(sample, mode, seg_num, seglen, short_size, target_size, img_mean, img_std):
            pickle_path = sample[0]
            try:
                if python_ver < (3, 0): # 根据python 版本选择不同读取方式
                    data_loaded = pickle.load(open(pickle_path, 'rb'))
                else:
                    data_loaded = pickle.load( open(pickle_path, 'rb'), encoding='bytes')

                vid, label, frames = data_loaded



                if len(frames) < 1:
                    logger.error('{} frame length {} less than 1.'.format( pickle_path, len(frames)))
                    return None, None
            except:
                logger.info('Error when loading {}'.format(pickle_path))
                return None, None

            if mode == 'train' or mode == 'valid' or mode == 'test':
                ret_label = label
            elif mode == 'infer':
                ret_label = vid

            # video_loader() 返回值为 seg_num*seglen 帧图片
            imgs = video_loader(frames, seg_num, seglen, mode)
            return imgs_transform(imgs, ret_label, mode, seg_num, seglen, short_size, target_size, img_mean, img_std)

        # 返回处理后的图像与对应的标签
        def imgs_transform(imgs, label, mode, seg_num, seglen, short_size, target_size, img_mean, img_std):
            imgs = group_scale(imgs, short_size)

            if mode == 'train':
                if self.name == "TSM":
                    imgs = group_multi_scale_crop(imgs, short_size)
                imgs = group_random_crop(imgs, target_size) ## 随机裁剪
                imgs = group_random_flip(imgs)              ## 随机翻转


                #添加数据增强部分，提升分类精度
                ## 作业中可添加其他的数据增强方式



            else:
                imgs = group_center_crop(imgs, target_size)

            # 将图像进行 reshape
            np_imgs = (np.array(imgs[0]).astype('float32').transpose((2, 0, 1))).reshape(1, 3, target_size, target_size) / 255

            for i in range(len(imgs) - 1):
                img = (np.array(imgs[i + 1]).astype('float32').transpose(
                    (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
                np_imgs = np.concatenate((np_imgs, img))
            imgs = np_imgs
            imgs -= img_mean
            imgs /= img_std


            ############################# 指定了返回的图像的 shape #########################
            # imgs = np.reshape(imgs, (seg_num, seglen * 3, target_size, target_size))
            imgs = np.reshape(imgs, (3, seglen * seg_num, target_size, target_size))

            return imgs, label

        def reader():
            with open(pickle_list) as flist:
                lines = [line.strip() for line in flist]
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    pickle_path = line.strip()
                    yield [pickle_path]


        # 上面都是自定的函数，下面语句才是在 _reader_creator() 中直接被调用的部分
        if format == 'pkl':
            decode_func = decode_pickle
        elif format == 'mp4':
            decode_func = decode_mp4
        else:
            raise "Not implemented format {}".format(format)

        mapper = functools.partial(
            decode_func,
            mode=mode,
            seg_num=seg_num,
            seglen=seglen,
            short_size=short_size,
            target_size=target_size,
            img_mean=img_mean,
            img_std=img_std)

        return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size)

# 多种裁剪方式
def group_multi_scale_crop(img_group, target_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
    scales = scales if scales is not None else [1, .875, .75, .66]
    input_size = [target_size, target_size]                     # 目标输出尺寸

    im_size = img_group[0].size                                 # 获取原图像 size

    # get random crop offset
    def _sample_crop_size(im_size):
        image_w, image_h = im_size[0], im_size[1]               # 原图的宽高

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in scales]       # 裁剪的目标尺寸为原图短边尺寸数值 * 裁剪比例（裁剪比例包含多个数） 
        crop_h = [
            input_size[1] if abs(x - input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            input_size[0] if abs(x - input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []                              # 存储多对 w 和 h 的值
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= max_distort:   # distort 形变，歪曲
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)        # 从存储的多对 w 和 h 值中随机选取一对
        # 是否为固定裁剪， fix 固定 
        # 不是固定的选择中的裁剪方式，即左上角起始点的坐标从可选的范围内进行随机选择
        if not fix_crop:   
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        # 几种固定起始点坐标的裁剪方式
        else:
            w_step = (image_w - crop_pair[0]) / 4
            h_step = (image_h - crop_pair[1]) / 4

            ret = list()    # ret 代表的是矩形框
            ret.append((0, 0))                          # upper left    左上
            if w_step != 0:
                ret.append((4 * w_step, 0))             # upper right   右上
            if h_step != 0:
                ret.append((0, 4 * h_step))             # lower left    左下
            if h_step != 0 and w_step != 0:
                ret.append((4 * w_step, 4 * h_step))    # lower right   右下
            if h_step != 0 or w_step != 0:
                ret.append((2 * w_step, 2 * h_step))    # center        中间

            if more_fix_crop:
                ret.append((0, 2 * h_step))           # center left
                ret.append((4 * w_step, 2 * h_step))  # center right
                ret.append((2 * w_step, 4 * h_step))  # lower center
                ret.append((2 * w_step, 0 * h_step))  # upper center

                ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

            w_offset, h_offset = random.choice(ret)

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
    crop_img_group = [
        img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        for img in img_group
    ]
    ret_img_group = [
        img.resize((input_size[0], input_size[1]), Image.BILINEAR)  # Image.BILINEAR 双线性差值
        for img in crop_img_group
    ]

    return ret_img_group

# 随机裁剪：结果为边长为target_size的正方形
def group_random_crop(img_group, target_size):
    w, h = img_group[0].size            # 获取图像的宽高 
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images

# 是否进行左右镜像翻转
def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group

# 中心裁剪，结果为边长均为 target_size 的正方形
def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size),  "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop

# 对图像进行resize，短边 ：长边 = 4 : 3 
def group_scale(imgs, target_size):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        # 较短的边长与目标尺寸相等
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs

# 读取一张图片，并转换为 RGB 格式，然后放入缓存中
def imageloader(buf):
    if isinstance(buf, str):
        img = Image.open(buf)
    else:
        img = Image.open(BytesIO(buf))

    return img.convert('RGB')

# 读取视频格式数据，返回值：len 为  segnum * seglen 张图片
def video_loader(frames, nsample, seglen, mode):
    videolen = len(frames)
    average_dur = int(videolen / nsample)

    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - seglen) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = frames[int(jj % videolen)]
            img = imageloader(imgbuf)
            imgs.append(img)

    return imgs 

# 读取mp4格式数据
def mp4_loader(filepath, nsample, seglen, mode):
    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)
    average_dur = int(len(sampledFrames) / nsample)
    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - 1) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = sampledFrames[int(jj % len(sampledFrames))]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)

    return imgs
