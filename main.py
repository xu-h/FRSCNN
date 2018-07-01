import numpy as np
import tensorflow as tf
from pathlib import WindowsPath as Path
import time
from PIL import Image
from matplotlib import pyplot as plt
import scipy

from config import sr_conf
from preprocess import crop_sub_images

def prelu(_x):
    alphas = tf.get_variable(
        'alpha', 
        1,
        initializer=tf.random_normal_initializer(0.0, 0.001),
        dtype=tf.float32
    )
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

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

def FSRCNN(train_data, train_label, test_data, test_label, is_full_img_test, scale, train_step):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7

    config = tf.estimator.RunConfig(
        model_dir = './model/FSRCNN',
        session_config=sess_config
    )
    estimator = tf.estimator.Estimator(
        model_fn=FSRCNN_fn,
        config=config,
        params={
            'is_full_img_test': is_full_img_test,
            'batch_size': 64,
            'in_size': train_data.shape[-1],
            'out_size': train_label.shape[-1],
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

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={ 'x': train_data.astype(np.float32) },
        y=train_label.astype(np.float32),
        batch_size=64,
        num_epochs=None,
        shuffle=True
    )

    start_time = time.time()
    # estimator.train(input_fn=train_input_fn, steps=train_step)
    end_time = time.time()

    if is_full_img_test:
        for i in range(len(test_data)):
            # make prediction for every image
            pre_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={
                    'x': test_data[i].astype(np.float32),
                    'y': test_label[i].astype(np.float32),
                },
                batch_size=1,
                num_epochs=1,
                shuffle=False
            )
            
            result_path = Path(f'./result-{scale}')
            if not result_path.exists():
                result_path.mkdir()

            predict_result = estimator.predict(input_fn=pre_input_fn)
            for result in predict_result:
                scipy.misc.imsave(result_path / f'{i}.png', result['labels'].squeeze())
                # cnn result
                psnr = round(float(result['cnn_psnr']), 4)
                ssim = round(float(result['cnn_ssim']), 4)
                scipy.misc.imsave(result_path / f'{i}-{psnr}-{ssim}.png', result['cnn_img'].squeeze()) 
                # bicubic result
                psnr = round(float(result['bi_psnr']), 4)
                ssim = round(float(result['bi_ssim']), 4)
                scipy.misc.imsave(result_path / f'{i}-bicubic-{psnr}-{ssim}.png', result['bi_img'].squeeze())

    else:
        pre_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                'x': test_data.astype(np.float32),
                'y': test_label.astype(np.float32),
            },
            batch_size=1,
            num_epochs=1,
            shuffle=False
        )

        predict_result = estimator.predict(input_fn=pre_input_fn)
        for result in predict_result:
            print(result)
            break
        
    print('time', end_time - start_time)

if __name__ == '__main__':
    is_full_img_test = True
    test_set = 'Set14'
    scale = 2
    train_step = 300000

    # preprocess data
    train_data, train_label = crop_sub_images('91-image', f'91-image-sub-{scale}', scale)
    # train_data = np.load(Path(sr_conf['dataset']) / '91-image-sub' / 'sub_img_data.npy')
    # train_label = np.load(Path(sr_conf['dataset']) / '91-image-sub' / 'sub_img_label.npy')
    print('train data: %d x (%d, %d)' % train_data.shape)
    print('train label: %d x (%d, %d)' % train_label.shape)

    if is_full_img_test:
        test_data_dir = Path(sr_conf['dataset']) / test_set / f'image_SRF_{scale}'
        test_data = []
        for img_path in test_data_dir.glob('*_LR.png'):
            img = Image.open(img_path).convert('L')
            w, h = img.size
            img = np.array(img.getdata()).reshape((1, h, w)) / 255
            test_data.append(img)
            print(f'test LR img: {img_path.name} {img.shape[1]}x{img.shape[2]}')
        test_label = []
        for img_path in test_data_dir.glob('*_HR.png'):
            img = Image.open(img_path).convert('L')
            w, h = img.size
            img = np.array(img.getdata()).reshape((1, h, w)) / 255
            test_label.append(img)
            print(f'test HR img: {img_path.name} {img.shape[1]}x{img.shape[2]}')
    else:
        # TODO refactor
        test_data = np.load(Path(sr_conf['dataset']) / 'Set5/image_SRF_2_sub/sub_img_data.npy')
        test_label = np.load(Path(sr_conf['dataset']) / 'Set5/image_SRF_2_sub/sub_img_label.npy')
        print('test data: %d x (%d, %d)' % test_data.shape)
        print('test label: %d x (%d, %d)' % test_label.shape)

    FSRCNN(train_data, train_label, test_data, test_label, is_full_img_test, scale, train_step)