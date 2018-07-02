import time
from pathlib import WindowsPath as Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import config
from preprocess import preprocess
from utils import load_dataset, prelu, save_prediction


def FSRCNN_fn(features, labels, mode, params):
    in_shape = features['x'].get_shape()
    tf.logging.warn('input shape')
    tf.logging.warn(in_shape)

    # reshape image before operatioh
    batch = params['batch_size']
    in_size, out_size = params['in_size'], params['out_size']
    if mode == tf.estimator.ModeKeys.PREDICT:
        out_shape = features['y'].get_shape()
        if params['is_full_img_test']:
            input_layer = tf.reshape(features['x'], (1, in_shape[1], in_shape[2], 1))
            labels = tf.reshape(features['y'], (1, out_shape[1], out_shape[2], 1))
        else:
            input_layer = tf.reshape(features['x'], (1, in_size, in_size, 1))
            labels = tf.reshape(features['y'], (1, out_size, out_size, 1))
    else:
        input_layer = tf.reshape(features['x'], (batch, in_size, in_size, 1))
        labels = tf.reshape(labels, (batch, out_size, out_size, 1))

    feature_extraction = tf.layers.conv2d(
        inputs=input_layer,
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

    # for i in range(6):
    #     tf.summary.image('conv1-%d' % i, c1[:, :, :, i:i + 1], 1)

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
        kernel_size=params['f2'],
        padding='same',
        activation=prelu,
        name='expanding'
    )

    # for i in range(16):
    #     tf.summary.image('conv2-%d' % i, c3[:, :, :, i:i + 1], 1)

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
    normal_labels = labels * 255

    psnr = tf.image.psnr(result_img, normal_labels, 255)
    ssim = tf.image.ssim(result_img, normal_labels, 255)
    tf.summary.scalar('psnr', tf.reduce_mean(psnr))
    tf.summary.scalar('ssim', tf.reduce_mean(ssim))
    
    loss = tf.losses.mean_squared_error(labels=labels, predictions=deconvolution)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.warn('predict shape')
        tf.logging.warn(result_img)
        bicubic_img = tf.image.resize_bicubic(
            input_layer * 255,
            (out_shape[1], out_shape[2])
        )

        return tf.estimator.EstimatorSpec(
            mode=mode, 
            predictions={
                'input': bicubic_img,
                'labels': labels,
                'bi_img': bicubic_img,
                'bi_psnr': tf.image.psnr(bicubic_img, normal_labels, 255),
                'bi_ssim': tf.image.ssim(bicubic_img, normal_labels, 255),
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
    else:
        # eval mode
        tf.logging.warn('result and normal labels')
        tf.logging.warn(result_img[0].shape)
        tf.logging.warn(normal_labels[0].shape)

        psnr = tf.image.psnr(result_img, normal_labels, 255)
        ssim = tf.image.ssim(result_img, normal_labels, 255)
            
        eval_metric_ops = {
            "psnr": psnr,
            "ssim": ssim
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

class FSRCNN(object):
    def __init__(self, model_dir, scale):
        # allow tf alloc more gpu memory
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7

        config = tf.estimator.RunConfig(
            model_dir = './model/FSRCNN',
            session_config=sess_config
        )
        self.estimator = tf.estimator.Estimator(
            model_fn=FSRCNN_fn,
            config=config,
            params={
                'f1': 5,        # kernel size of feature extraction
                'd': 64,        # feature maps number of feature extraction
                'f2': 1,        # kernel size of shrinking
                's': 16,        # feature maps number of shrinking
                'f3': 3,        # mapping layer kernel size
                'm': 5,         # mapping layer number
                'k': scale,          # scale factor,
                'f5': 9,        # kernel size of deconvolution
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
    train_step = 300000 

    # train
    train_data, train_label = preprocess(config.train_dataset, False, 2)
    print('train data: %d x (%d, %d)' % train_data.shape)
    print('train label: %d x (%d, %d)' % train_label.shape)
    
    model = FSRCNN('model/FSRCNN1', scale)
    # model.train(train_data, train_label, config.batch_size, train_step)

    # test
    for test_dataset in config.test_dataset:
        test_data, test_label = load_dataset(test_dataset, scale)
        for lr_img, hr_img in zip(test_data, test_label):
            print(lr_img.shape, hr_img.shape)
            # result = model.predict(lr_img, hr_img)
            # save_prediction(result)
