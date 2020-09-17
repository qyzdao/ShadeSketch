"""
ShadeSketch
https://github.com/qyzdao/ShadeSketch

Learning to Shadow Hand-drawn Sketches
Qingyuan Zheng, Zhuoru Li, Adam W. Bargteil

Copyright (C) 2020 The respective authors and Project HAT. All rights reserved.
Licensed under MIT license.
"""

import os
import numpy as np
import tensorflow as tf
from model import build_generator, build_discriminator
from utils import load_data, process_data
import matplotlib.pyplot as plt
import datetime

# import keras
keras = tf.keras
K = keras.backend
Model = keras.models.Model
Input = keras.layers.Input
Adam = keras.optimizers.Adam

# ---------------------
#  Train Config
# ---------------------
SEED = 1
BATCH_SIZE = 8
ITERATIONS = 50000


def data_generator(batch_size, seed):
    # Our dataset is small, we can pack it as numpy, then load all data into memory
    # Line, Cond(label), Shade
    x_data, c_data, y_data = load_data('./data.npy')
    print('Load {} data pairs'.format(len(x_data)))

    counts = 0
    while True:
        np.random.seed(seed + counts)
        idx = np.random.randint(0, x_data.shape[0], batch_size)

        x_batch, c_batch, p_batch, y_batch = process_data(x_data[idx], c_data[idx], y_data[idx], seed=(seed + counts))

        counts += batch_size

        # Line, Cond(label), Pos, Shade
        yield x_batch, c_batch, p_batch, y_batch


def plot_figs(x_batch, y_batch, cond_batch, result, combine, name):
    imgs = result[:3]
    output = np.concatenate([x_batch[:3], (imgs + 1) / 2, y_batch[:3]])
    output = (1 - output) * 255

    if combine:
        output[3] = output[3] * 0.2 + output[0] * 0.8
        output[4] = output[4] * 0.2 + output[1] * 0.8
        output[5] = output[5] * 0.2 + output[2] * 0.8

        output[6] = output[6] * 0.2 + output[0] * 0.8
        output[7] = output[7] * 0.2 + output[1] * 0.8
        output[8] = output[8] * 0.2 + output[2] * 0.8

    r, c = 3, 3
    titles = ['Input', 'Output', 'Target']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(output[cnt].squeeze(), cmap='gray')
            axs[i, j].set_title(titles[i] + ' (%d,%d,%d)' % tuple(cond_batch[:3][j]))
            axs[i, j].axis('off')
            cnt += 1

    fig.savefig('./output/%s.png' % name, dpi=250)
    plt.close()


def train():
    if not os.path.exists('./output'):
        os.makedirs('./output')

    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    # ---------------------
    #  Model and Optimizer
    # ---------------------
    print('Building models...')

    baseG = build_generator()
    baseD = build_discriminator()

    # Build D train
    rp_in = Input(shape=(3,))
    rl_in = Input(shape=(None, None, 1))
    rs_in = Input(shape=(None, None, 1))

    fp_in = Input(shape=(3,))
    fl_in = Input(shape=(None, None, 1))

    G = Model(inputs=baseG.inputs, outputs=baseG.outputs)
    D = Model(inputs=baseD.inputs, outputs=baseD.outputs)

    G.trainable = False
    D.trainable = True

    fs_1, fs_2, fs_3 = G([fp_in, fl_in])

    r_valid = D([rp_in, rl_in, rs_in])
    f_valid = D([fp_in, fl_in, fs_1])

    D_train = Model(
        [rp_in, fp_in, rl_in, fl_in, rs_in],
        [r_valid, f_valid]
    )

    # Adv Loss (Vanilla GAN)
    D_train.add_loss(0.5 * (K.mean(-K.log(r_valid + 1e-9) - K.log(1 - f_valid + 1e-9))))

    D_train.compile(optimizer=Adam(2e-4, 0.0, 0.9))

    # Build G train
    fp_in = Input(shape=(3,))
    fl_in = Input(shape=(None, None, 1))
    fs_in = Input(shape=(None, None, 1))

    G = Model(inputs=baseG.inputs, outputs=baseG.outputs)
    D = Model(inputs=baseD.inputs, outputs=baseD.outputs)

    G.trainable = True
    D.trainable = False

    fs_1, fs_2, fs_3 = G([fp_in, fl_in])

    f_valid = D([fp_in, fl_in, fs_1])

    G_train = Model(
        [fp_in, fl_in, fs_in],
        [fs_1, fs_2, fs_3, f_valid]
    )

    # MSE Loss + TV Reg for main output
    G_train.add_loss(5e-1 * K.mean(K.square(fs_1 - fs_in)) + 1e-6 * tf.reduce_sum(tf.image.total_variation(fs_1)))
    # MSE Loss for sub output
    G_train.add_loss(2e-1 * (K.mean(K.square(fs_2 - fs_in)) + K.mean(K.square(fs_3 - fs_in))))
    # Adv Loss for main output (Vanilla GAN)
    G_train.add_loss(4e-1 * K.mean(-K.log(f_valid + 1e-9)))

    G_train.compile(optimizer=Adam(2e-4, 0.0, 0.9))
    # ---------------------
    #  Train
    # ---------------------
    print('Start Training...')

    start_time = datetime.datetime.now()
    train_generator = data_generator(batch_size=BATCH_SIZE, seed=SEED)

    for iteration in range(1, ITERATIONS + 1):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        l_train, _, p_train, s_train = next(train_generator)

        # Train the discriminator
        D_loss = D_train.train_on_batch(
            [p_train, p_train, l_train, l_train, s_train],
            None
        )

        # ---------------------
        #  Train Generator
        # ---------------------
        l_train, _, p_train, s_train = next(train_generator)

        # Train the generator
        G_loss = G_train.train_on_batch(
            [p_train, l_train, s_train],
            None
        )

        print(
            '[Time: %s] [Iteration: %d] [D loss: %f] [G loss: %f]' %
            (
                datetime.datetime.now() - start_time,
                iteration,
                D_loss,
                G_loss,
            )
        )

        # Save training samples
        if iteration % 100 == 0:
            # At least 3 images, bs < 3 causes error
            train_x_batch, cond_batch, train_pos_batch, train_y_batch = next(train_generator)
            gen_imgs, s1, s2 = G.predict([train_pos_batch, train_x_batch])

            plot_figs(train_x_batch, train_y_batch, cond_batch, gen_imgs, True, '%d' % iteration)
            plot_figs(train_x_batch, train_y_batch, cond_batch, s1, False, '%d_s1' % iteration)
            plot_figs(train_x_batch, train_y_batch, cond_batch, s2, False, '%d_s2' % iteration)

        if iteration % 200 == 0:
            D.save_weights('./weights/G_%05d.h5' % iteration)
            G.save_weights('./weights/D_%05d.h5' % iteration)


if __name__ == '__main__':
    train()
