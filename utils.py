"""
ShadeSketch
https://github.com/qyzdao/ShadeSketch

Learning to Shadow Hand-drawn Sketches
Qingyuan Zheng, Zhuoru Li, Adam W. Bargteil

Copyright (C) 2020 The respective authors and Project HAT. All rights reserved.
Licensed under MIT license.
"""

import os
import csv
import scipy.ndimage as ndi
import numpy as np
import cv2


def norm_line(data_dir, model_dir):
    import tensorflow

    if hasattr(tensorflow.compat, 'v1'):
        tf = tensorflow.compat.v1
        tf.disable_v2_behavior()
    else:
        tf = tensorflow

    with tf.gfile.FastGFile(model_dir, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='lineNorm')

    if not os.path.exists(os.path.join(data_dir, 'norm')):
        os.makedirs(os.path.join(data_dir, 'norm'))

    with tf.Session() as sess:
        for root, dirs, files in os.walk(os.path.join(data_dir, 'line'), topdown=False):
            for name in files:
                line_path = os.path.join(root, name)
                print(line_path)

                img = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
                tensors = np.reshape(img, (1, img.shape[0], img.shape[1], 1)).astype(np.float32) / 255.

                tensors = sess.run('lineNorm/conv2d_9/Sigmoid:0', {'lineNorm/input_1:0': tensors})

                cv2.imwrite(os.path.join(data_dir, 'norm', name), np.squeeze(tensors) * 255.)


def pack_data(data_dir, output_dir='./data.npy', use_norm=True, norm_model='./models/linenorm.pb'):
    linetype = 'norm' if use_norm else 'line'

    if use_norm:
        assert norm_model is not None, "Please assign a line normalization model."
        norm_line(data_dir, norm_model)

    with open(os.path.join(data_dir, 'anno.csv'), 'r') as f:
        lines = list(csv.reader(f, delimiter=','))

    data = []
    for l in lines:
        file_index = l[0]

        # The line in the dataset is not normalized, please normalize it followed the paper
        line = cv2.imread(os.path.join(data_dir, linetype, '%s.png' % file_index), cv2.IMREAD_GRAYSCALE)
        shade = cv2.imread(os.path.join(data_dir, 'shade', '%s.png' % file_index), cv2.IMREAD_GRAYSCALE)
        # Mask is not used
        # mask = cv2.imread(os.path.join(data_dir, 'mask', '%s.png' % file_index), cv2.IMREAD_GRAYSCALE)
        cond = np.array([int(i) for i in l[1:]], np.int)

        data.append([line, cond, shade])

    np.save(output_dir, np.array(data))


def load_data(filename):
    # Line, Cond(label), Shade
    x, c, y = [], [], []

    f = np.load(filename, allow_pickle=True)

    for e in f:
        x.append(np.expand_dims(e[0], axis=-1))
        c.append(e[1])
        y.append(np.expand_dims(e[2], axis=-1))

    # Line, Cond(label), Shade
    return np.array(x), np.array(c), np.array(y)


def cond_to_pos(cond):
    # Convert the user label lighting direction to position
    cond_pos_rel = {
        '002': [0, 0, -1],
        '110': [0, 1, -1], '210': [1, 1, -1], '310': [1, 0, -1], '410': [1, -1, -1], '510': [0, -1, -1],
        '610': [-1, -1, -1], '710': [-1, 0, -1], '810': [-1, 1, -1],
        '120': [0, 1, 0], '220': [1, 1, 0], '320': [1, 0, 0], '420': [1, -1, 0], '520': [0, -1, 0],
        '620': [-1, -1, 0], '720': [-1, 0, 0], '820': [-1, 1, 0],
        '130': [0, 1, 1], '230': [1, 1, 1], '330': [1, 0, 1], '430': [1, -1, 1], '530': [0, -1, 1],
        '630': [-1, -1, 1], '730': [-1, 0, 1], '830': [-1, 1, 1],
        '001': [0, 0, 1]
    }

    cond_str = ''.join([str(i) for i in cond])

    return cond_pos_rel[cond_str]


def normalize_angle(rg):
    rg = rg % 360

    if rg < 0:
        rg += 360

    return rg


def process_data(x_batch, cond_batch, y_batch, seed):
    rot = [90, 45, 0, 315, 270, 225, 180, 135]
    idx = [1, 2, 3, 4, 5, 6, 7, 8]
    r2i = dict(zip(rot, idx))
    i2r = dict(zip(idx, rot))

    pos_batch = []

    for i in range(len(x_batch)):

        do_rotation = True
        if cond_batch[i][0] == 0:
            do_rotation = False

        # Data augment
        # Rotation -10 ~ 10, Shift -0.2 ~ 0.2, Zoom 0.9 ~ 1.1
        x_batch[i] = random_transform(x_batch[i], seed + i)
        y_batch[i] = random_transform(y_batch[i], seed + i)

        if do_rotation:
            angle = i2r[cond_batch[i][0]]

            np.random.seed(seed + i)
            rot_angle = np.random.randint(0, 7) * 45

            if not rot_angle == 0:
                x_batch[i] = rotation(x_batch[i], -rot_angle)
                y_batch[i] = rotation(y_batch[i], -rot_angle)

                cond_batch[i][0] = r2i[normalize_angle(angle + rot_angle)]

        pos_batch.append(cond_to_pos(cond_batch[i]))

    pos_batch = np.array(pos_batch).astype(np.float32)
    cond_batch = np.array(cond_batch).astype(np.float32)
    train_x_batch = np.clip(1 - x_batch.astype(np.float32) / 255., 0., 1.)
    train_y_batch = 2 * np.clip(1 - y_batch.astype(np.float32) / 255., 0., 1.) - 1

    return train_x_batch, cond_batch, pos_batch, train_y_batch


"""
Implementation of data augment based on Keras's image preprocessing module

Keras

MIT License
COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015, Google, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

https://github.com/keras-team/keras/blob/keras-2/keras/preprocessing/image.py
"""


def random_transform(x, seed=None):
    """Randomly augment a single image tensor.

    # Arguments
        x: 3D tensor, single image.
        seed: random seed.

    # Returns
        A randomly transformed version of the input (same shape).
    """
    np.random.seed(seed)

    img_row_axis = 0
    img_col_axis = 1
    img_channel_axis = 2

    rotation_range = 10
    theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))

    height_shift_range = width_shift_range = 0.2
    if height_shift_range:
        try:  # 1-D array-like or int
            tx = np.random.choice(height_shift_range)
            tx *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            tx = np.random.uniform(-height_shift_range,
                                   height_shift_range)
        if np.max(height_shift_range) < 1:
            tx *= x.shape[img_row_axis]
    else:
        tx = 0

    if width_shift_range:
        try:  # 1-D array-like or int
            ty = np.random.choice(width_shift_range)
            ty *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            ty = np.random.uniform(-width_shift_range,
                                   width_shift_range)
        if np.max(width_shift_range) < 1:
            ty *= x.shape[img_col_axis]
    else:
        ty = 0

    zoom_range = (0.9, 1.1)
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis, fill_mode='nearest')

    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel
                      in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def rotation(x, rg):
    theta = np.deg2rad(rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, 2, fill_mode='constant', cval=255)

    return x


if __name__ == '__main__':
    pack_data('./ShadeSketchDataset')
