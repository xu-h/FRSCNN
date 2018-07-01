import os
import sys
import glob
import numpy as np
from PIL import Image
import pdb
from pathlib import WindowsPath as Path
import math
import matplotlib.pyplot as plt

from config import sr_conf


def expand(origin, expand):
    """Expand dataset by scaling and rotation."""
    dataset = Path(sr_conf['dataset'])
    origin_dir = dataset / origin
    expand_dir = dataset / expand

    # check path
    if not origin_dir.exists():
        print('Path does not exit.')
        return
    if not expand_dir.exists():
        expand_dir.mkdir()

    # expand
    for img_path in origin_dir.glob('*.bmp'):
        img = Image.open(img_path)
        w, h = img.size
        cnt = 0
        for scale in (1, 0.9, 0.8, 0.7, 0.6):
            new_w, new_h = math.floor(w * scale), math.floor(h * scale)
            img1 = img.resize((new_w, new_h), Image.BICUBIC)
            for rotation in (0, 90, 180, 270):
                img2 = img1.rotate(rotation, expand=True)
                img2.save(expand_dir / f'{img_path.stem}-{cnt}{img_path.suffix}')
                cnt += 1

def crop_sub_images(origin, sub_dir, scale):
    dataset = Path(sr_conf['dataset'])
    origin_dir = dataset / origin
    sub_dir = dataset / sub_dir

    # check path
    if not origin_dir.exists():
        print('Path does not exit.')
        return
    if not sub_dir.exists():
        sub_dir.mkdir()

    data = []
    label = []
    
    glob = list(origin_dir.glob('*.bmp'))
    if len(glob) == 0:
        glob = list(origin_dir.glob('*_HR.png'))
    for img_path in glob:
        # generate sub imgs for specified scale factor
        img = Image.open(img_path).convert('L') # convert: convert to grayscale
        w, h = img.size
        scale_w, scale_h = w // scale, h // scale
        w, h = scale_w * scale, scale_h * scale # round the width and height
        img = img.crop((0, 0, w, h))
        scale_img = img.resize((scale_w, scale_h), Image.BICUBIC)
        
        # conver to numpy
        img = np.array(img.getdata()).reshape((h, w)) / 255
        scale_img = np.array(scale_img.getdata()).reshape((scale_h, scale_w)) / 255
        # for debug
        # plt.figure('HR')
        # plt.imshow(img, cmap='gray')
        # plt.figure('SR')
        # plt.imshow(scale_img, cmap='gray')
        # plt.show()

        # generate sub imgs
        scale_size, img_size = sr_conf['sub_img_size'][scale]

        for scale_y in range(0, scale_h - scale_size, scale):
            for scale_x in range(0, scale_w - scale_size, scale):
                data.append(scale_img[scale_y:scale_y + scale_size, scale_x:scale_x + scale_size])
                x = scale_x * scale
                y = scale_y * scale
                label.append(img[y:y + img_size, x:x + img_size])
                # print(data[-1].shape, label[-1].shape)
    data = np.stack(data)
    label = np.stack(label)
    print(data.shape, label.shape)
    npy_path = sub_dir / 'sub_img_data.npy'
    with npy_path.open('wb') as f:
        np.save(f, data)
    npy_path = sub_dir / 'sub_img_label.npy'
    with npy_path.open('wb') as f:
        np.save(f, label)
    return data, label


if __name__ == '__main__':
    if len(sys.argv) == 3:
        crop_sub_images(sys.argv[1], sys.argv[2], 2)
    elif len(sys.argv) == 4:
        expand(sys.argv[1], sys.argv[2])
        crop_sub_images(sys.argv[2], sys.argv[3], 2)
    else:
        print("Missing argument: You must specify a folder with images to expand")
