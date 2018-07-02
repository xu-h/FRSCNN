import time
from pathlib import WindowsPath as Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import config
from preprocess import preprocess
from utils import load_dataset, prelu, save_result
from pprint import pprint


def FSRCNN_fn(features, labels, mode, params):
    # reshape image before operatioh
    in_shape = features['x'].get_shape()
    if mode == tf.estimator.ModeKeys.PREDICT:
        batch = 1
        labels = features['y']
    else:
        batch = in_shape[0]
    data = tf.reshape(features['x'], (batch, in_shape[1], in_shape[2], 1))
    out_shape = labels.get_shape()
    labels = tf.reshape(labels, (batch, out_shape[1], out_shape[2], 1))
    # if not normalize, loss may be NaN
    normal_data = data / 255
    normal_labels = labels / 255

    tf.logging.info('input shape')
    tf.logging.info(in_shape)
    tf.logging.info('output shape')
    tf.logging.info(out_shape)

    feature_extraction = tf.layers.conv2d(
        inputs=normal_data,
        filters=params['d'],
        kernel_size=params['f1'], # same value at two dims
        padding='same',
        activation=prelu,
        name='feature_extraction'
    )

    shrinking = tf.layers.conv2d(
        inputs=feature_extraction,
        filters=params['s'],
        kernel_size=params['f2'],
        padding='same',
        activation=prelu,
        name='shrinking'
    )

    mapping = shrinking
    for i in range(params['m']):
        mapping = tf.layers.conv2d(
            inputs=mapping,
            filters=params['s'],
            kernel_size=params['f3'],
            padding='same',
            activation=prelu,
            name=f'mapping-{i}'
        )

    expanding = tf.layers.conv2d(
        inputs=mapping,
        filters=params['d'],
        kernel_size=params['f4'],
        padding='same',
        activation=prelu,
        name='expanding'
    )

    deconvolution = tf.layers.conv2d_transpose(
        inputs=expanding,
        filters=1,
        kernel_size=params['f5'],
        strides=params['k'],
        padding="same",
        activation=None,
        name='deconvolution'
    )
    
    result_img = deconvolution * 255

    psnr = tf.image.psnr(result_img, labels, 255)
    ssim = tf.image.ssim(result_img, labels, 255)
    tf.summary.scalar('psnr', tf.reduce_mean(psnr))
    tf.summary.scalar('ssim', tf.reduce_mean(ssim))
    
    loss = tf.losses.mean_squared_error(labels=normal_labels, predictions=deconvolution)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        bicubic_img = tf.image.resize_bicubic(
            data,
            (out_shape[1], out_shape[2])
        )

        return tf.estimator.EstimatorSpec(
            mode=mode, 
            predictions={
                'input': bicubic_img,
                'labels': labels,
                'bi_img': bicubic_img,
                'bi_psnr': tf.image.psnr(bicubic_img, labels, 255),
                'bi_ssim': tf.image.ssim(bicubic_img, labels, 255),
                'cnn_img': result_img,
                'cnn_psnr': psnr,
                'cnn_ssim': ssim
            }
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "psnr": psnr,
            "ssim": ssim
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

class FSRCNN(object):
    def __init__(self, model_dir, scale):
        # allow tf alloc more gpu memory
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = 1

        config = tf.estimator.RunConfig(
            model_dir = model_dir,
            session_config=sess_config
        )
        self.estimator = tf.estimator.Estimator(
            model_fn=FSRCNN_fn,
            config=config,
            params={
                'k': scale,     # scale factor,
                'f1': 5,        # kernel size of feature extraction
                'f2': 1,        # kernel size of shrinking
                'f3': 3,        # kernel size of mapping layer
                'f4': 1,        # kernel size of expanding
                'f5': 9,        # kernel size of deconvolution
                'd': 64,        # feature maps number of feature extraction
                's': 16,        # feature maps number of shrinking
                'm': 5,         # mapping layer number
            }
        )

    def train(self, data, label, batch, step):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={ 'x': data.astype(np.float32) },
            y=train_label.astype(np.float32),
            batch_size=batch,
            num_epochs=None,
            shuffle=True
        )
        self.estimator.train(input_fn=train_input_fn, steps=step)

    def predict(self, data, label):
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                'x': data.astype(np.float32),
                'y': label.astype(np.float32),
            },
            batch_size=1,
            num_epochs=1,
            shuffle=False
        )
        return self.estimator.predict(input_fn=pred_input_fn)


if __name__ == '__main__':
    scale = 2
    train_step = 300
    model_dir = 'model/FSRCNN1'

    # train
    train_data, train_label = preprocess(config.train_dataset, False, 2)
    print('train data: %d x (%d, %d)' % train_data.shape)
    print('train label: %d x (%d, %d)' % train_label.shape)
    
    model = FSRCNN(model_dir, scale)
    model.train(train_data, train_label, config.batch_size, train_step)

    # test
    performance = {}
    for test_dataset in config.test_dataset:
        test_data, test_label = load_dataset(test_dataset, scale)

        result_dir = Path(f'result-{scale}-{test_dataset}')
        if not result_dir.exists():
            result_dir.mkdir()
        perf = []

        for i, (lr_img, hr_img) in enumerate(zip(test_data, test_label)):
            for result in model.predict(lr_img, hr_img):
                save_result(result, result_dir, i)
                perf.append((i, result['cnn_psnr'], result['cnn_ssim'], 
                    result['bi_psnr'], result['bi_ssim']))
        performance[test_dataset] = perf
        
    for test_dataset in config.test_dataset:
        print(f'Performance on {test_dataset}')
        print('no\tFSRCNN\tBicubic')
        for p in performance[test_dataset]:
            print(f'{p[0]: 2d}\t{p[1]:>.4f}\t{p[2]:.4f}\t{p[3]:>.4f}\t{p[4]:.4f}')

