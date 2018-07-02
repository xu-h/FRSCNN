import tensorflow as tf

def load_dataset(dataset):
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

def save_prediction():
    result_path = Path(f'./result-{scale}')
    if not result_path.exists():
        result_path.mkdir()

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

def progress_bar(cur, total):
    percent = cur / total
    bar_length = 20
    sybom_num = int(percent * bar_length)
    bar = '=' * sybom_num + ' ' * (bar_length - sybom_num)
    return f'[{bar}] {percent * 100:.2f}%'