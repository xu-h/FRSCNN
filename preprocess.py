"""Preprocess of dataset."""
import argparse
import glob
import math

import numpy as np
from PIL import Image

import config


def expand(img):
    """Expand dataset by scaling and rotation."""
    w, h = img.size
    img_list = []
    for scale in config.expand['scale']:
        new_w, new_h = math.floor(w * scale), math.floor(h * scale)
        img1 = img.resize((new_w, new_h), Image.BICUBIC)
        for rotation in config.expand['rotation']:
            img_list.append(img1.rotate(rotation, expand=True))
    return img_list

def crop_sub_images(img, scale):
    w, h = img.size
    scale_w, scale_h = w // scale, h // scale
    w, h = scale_w * scale, scale_h * scale # round the width and height
    img = img.crop((0, 0, w, h))
    scale_img = img.resize((scale_w, scale_h), Image.BICUBIC)
    
    # conver to numpy
    img = np.array(img.getdata()).reshape((h, w)) / 255
    scale_img = np.array(scale_img.getdata()).reshape((scale_h, scale_w)) / 255

    # generate sub imgs
    scale_size, img_size = config.sub_img_size[scale]
    data = []
    label = []

    for scale_y in range(0, scale_h - scale_size, scale):
        for scale_x in range(0, scale_w - scale_size, scale):
            data.append(scale_img[scale_y:scale_y + scale_size, scale_x:scale_x + scale_size])
            x = scale_x * scale
            y = scale_y * scale
            label.append(img[y:y + img_size, x:x + img_size])
            # print(data[-1].shape, label[-1].shape)
    data = np.stack(data)
    label = np.stack(label)
    return data, label

def preprocess(dataset, is_expand, scale):
    """Preprocess of dataset and generate numpy ndarray for train.

    Now is_expand must be False.
    If is_expand is True, the programe may used up all memory.

    :param dataset: An unix style pathname pattern for images.
    :param is_expand: If expand dataset by rotation and scaling.
    :return: (data, label)
    """
    data = []
    label = []

    for img_path in glob.glob(dataset):
        print(f'Processing {img_path}')
        img = Image.open(img_path).convert('L')
        if is_expand:
            imgs = expand(img)
        else:
            imgs = [img]
        for img in imgs:
            # crop an image into sub images
            data_slice, label_slice = crop_sub_images(img, scale)
            data.append(data_slice)
            label.append(label_slice)
    # concatenate sub images of all images
    data = np.concatenate(data)
    label = np.concatenate(label)
    return (data, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess of dataset.'
    )
    parser.add_argument('dataset', help="Choose dataset defined in config.py")
    parser.add_argument('--scale', type=int, help="Scale factor used for crop.")
    parser.add_argument('--expand', type=bool, default=False, help='Use rotation and subsampling expand the dataset.')

    args = parser.parse_args()

    # TODO After expanding the dataset, sub images consumes too much memory
    data, label = preprocess(config.dataset[args.dataset], False, args.scale)
    
    print(data.shape, label.shape)
    np.save('data.npy', data)
    np.save('label.npy', label)
