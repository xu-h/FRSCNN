"""Config for FSRCNN."""

dataset = {
    '91-image': '../dataset/91-image/*.bmp',
    'general-100': '../dataset/General-100/*.bmp',
    'set5': '../dataset/Set5/image_SRF_2/*HR.png',
    'set14': '../dataset/Set14/image_SRF_2/*HR.png',
}

expand = {
    'scale': (1, 0.9, 0.8, 0.7, 0.6),
    'rotation': (0, 90, 180, 270)
}

# input and output sub image size for different scale factor
sub_img_size = {
    2: (10, 20),
    3: (7, 21),
    4: (6, 24)
}

train_dataset = '91-image'
test_dataset = 'set5'
