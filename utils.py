import glob

import numpy as np
import tensorflow as tf
from PIL import Image
import scipy

import config


def progress_bar(cur, total):
    percent = cur / total
    bar_length = 25
    sybom_num = int(percent * bar_length)
    bar = '=' * sybom_num + ' ' * (bar_length - sybom_num)
    return f'[{bar}] {percent * 100:.2f}%'


def prelu(_x):
    alphas = tf.get_variable(
        'alpha',
        1,
        initializer=tf.random_normal_initializer(0.0, 0.001),
        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def load_dataset(dataset, scale):
    data = []
    label = []
    lr_glob, hr_glob = config.dataset[dataset]
    print(f'Loading images: {lr_glob} {hr_glob}')
    for img_path in glob.glob(lr_glob):
        img = Image.open(img_path).convert('L')
        w, h = img.size
        data.append(np.array(img.getdata()).reshape((1, h, w)))
    for img_path in glob.glob(hr_glob):
        img = Image.open(img_path).convert('L')
        w, h = img.size
        label.append(np.array(img.getdata()).reshape((1, h, w)))
    return data, label


def save_result(result, path, name):
    scipy.misc.imsave(path / f'{name}.png', result['labels'].squeeze())
    # cnn result
    psnr = round(float(result['cnn_psnr']), 4)
    ssim = round(float(result['cnn_ssim']), 4)
    scipy.misc.imsave(
        path / f'{name}-{psnr}-{ssim}.png',
        result['cnn_img'].squeeze()
    )
    # bicubic result
    psnr = round(float(result['bi_psnr']), 4)
    ssim = round(float(result['bi_ssim']), 4)
    scipy.misc.imsave(
        path / f'{name}-bicubic-{psnr}-{ssim}.png',
        result['bi_img'].squeeze()
    )
