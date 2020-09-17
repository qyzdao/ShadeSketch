"""
ShadeSketch
https://github.com/qyzdao/ShadeSketch

Learning to Shadow Hand-drawn Sketches
Qingyuan Zheng, Zhuoru Li, Adam W. Bargteil

Copyright (C) 2020 The respective authors and Project HAT. All rights reserved.
Licensed under MIT license.
"""

import tensorflow as tf
from layers import PixelwiseConcat, SelfAttention, SubPixelConv2D, CoordinateChannel2D, Composite

# import keras
keras = tf.keras
K = keras.backend
Model = keras.models.Model
Input = keras.layers.Input
Conv2D = keras.layers.Conv2D
BatchNormalization = keras.layers.BatchNormalization
Add = keras.layers.Add
Concatenate = keras.layers.Concatenate
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
GlobalAvgPool2D = keras.layers.GlobalAvgPool2D
UpSampling2D = keras.layers.UpSampling2D
Multiply = keras.layers.Multiply
LeakyReLU = keras.layers.LeakyReLU

IMG_SHAPE = (320, 320, 1)
IMG_HEIGHT, IMG_WIDTH, IMG_CHAN = IMG_SHAPE


def residual_block_downscaling(input_tensor, filters, strides=2):
    filter1, filter2, filter3 = filters

    x = BatchNormalization()(input_tensor)
    x = LeakyReLU()(x)
    x = Conv2D(filter1, 1, use_bias=False)(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filter2, 3, strides=strides, padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filter3, 1, use_bias=False)(x)

    shortcut = Conv2D(filter3, 1, strides=strides, use_bias=False)(input_tensor)

    x = Add()([x, shortcut])

    return x


def residual_block_upscaling(input_tensor, filters, strides=2):
    filter1, filter2, filter3 = filters

    x = BatchNormalization()(input_tensor)
    x = LeakyReLU()(x)
    x = Conv2D(filter1, 1, use_bias=False)(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = SubPixelConv2D(filter2, 3, strides, padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filter3, 1, use_bias=False)(x)

    shortcut = SubPixelConv2D(filter3, 1, strides, use_bias=False)(input_tensor)

    x = Add()([x, shortcut])
    x = Dropout(0.1)(x)

    return x


def residual_block(input_tensor, filters, shortcut_conv=False):
    filter1, filter2, filter3 = filters

    x = BatchNormalization()(input_tensor)
    x = LeakyReLU()(x)
    x = Conv2D(filter1, 1, use_bias=False)(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filter2, 3, padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filter3, 1, use_bias=False)(x)

    if shortcut_conv:
        shortcut = Conv2D(filter3, 1, use_bias=False)(input_tensor)
        x = Add()([x, shortcut])
    else:
        x = Add()([x, input_tensor])

    return x


def cse_block(l, h, ratio):
    mean = GlobalAvgPool2D()(l)

    cse = Dense(K.int_shape(h)[3] // ratio, use_bias=False)(mean)
    cse = LeakyReLU()(cse)
    cse = Dense(K.int_shape(h)[3], activation='sigmoid', use_bias=False)(cse)

    return Multiply()([h, cse])


def sse_block(l, h):
    sse = Conv2D(1, 1, activation='sigmoid', use_bias=False)(l)

    return Multiply()([h, sse])


def se(l, h, ratio=2):
    cse = cse_block(l, h, ratio)
    sse = sse_block(l, h)

    return Add()([cse, sse])


def film(input_cond, input_tensor, filters):
    gamma = Dense(filters)(input_cond)
    beta = Dense(filters)(input_cond)

    return Add()([Multiply()([gamma, input_tensor]), beta])


def filmed_residual_block(input_cond, input_tensor, filters):
    a = Conv2D(filters, 1, use_bias=False)(input_tensor)
    a = LeakyReLU()(a)

    b = Conv2D(filters, 3, padding='same', use_bias=False)(a)
    b = BatchNormalization(scale=False, center=False)(b)
    b = film(input_cond, b, filters)
    b = LeakyReLU()(b)

    return Add()([a, b])


def build_generator():
    input_cond = Input((3,))
    embed_cond = Dense(128, activation='tanh')(input_cond)
    input_img = Input((None, None, IMG_CHAN))

    # encoder
    d1 = CoordinateChannel2D()(input_img)
    d1 = residual_block(d1, (8, 8, 32), True)  # 1/1 320
    d1 = residual_block(d1, (8, 8, 32))

    d2 = residual_block_downscaling(d1, (16, 16, 64))  # 1/2 160
    d2 = residual_block(d2, (16, 16, 64))
    d2 = residual_block(d2, (16, 16, 64))

    d3 = residual_block_downscaling(d2, (32, 32, 128))  # 1/4 80
    d3 = residual_block(d3, (32, 32, 128))
    d3 = residual_block(d3, (32, 32, 128))

    d4 = residual_block_downscaling(d3, (64, 64, 256))  # 1/8 40
    d4 = residual_block(d4, (64, 64, 256))
    d4 = residual_block(d4, (64, 64, 256))

    d5 = residual_block_downscaling(d4, (64, 64, 256))  # 1/16 20
    d5 = residual_block(d5, (64, 64, 256))
    d5 = residual_block(d5, (64, 64, 256))

    # bottleneck
    d6 = residual_block_downscaling(d5, (128, 128, 512))  # 1/32 10
    d6 = residual_block(d6, (128, 128, 512))
    d6 = residual_block(d6, (128, 128, 512))

    d6 = filmed_residual_block(embed_cond, CoordinateChannel2D()(d6), 512)
    d6 = residual_block(d6, (128, 128, 512))
    d6 = residual_block(d6, (128, 128, 512))
    d6 = residual_block(d6, (128, 128, 512))
    d6 = residual_block(d6, (128, 128, 512))
    d6 = SelfAttention()(d6)

    # decoder
    u1 = residual_block_upscaling(d6, (64, 64, 256))  # 20
    u1 = CoordinateChannel2D()(Concatenate()([u1, se(u1, d5)]))
    u1 = filmed_residual_block(embed_cond, u1, 256)
    u1 = residual_block(u1, (64, 64, 256))
    u1 = residual_block(u1, (64, 64, 256))
    u1 = SelfAttention()(u1)

    s1 = UpSampling2D(16)(Conv2D(1, 1, activation='tanh')(u1))

    u2 = residual_block_upscaling(u1, (64, 64, 256))  # 40
    u2 = CoordinateChannel2D()(Concatenate()([u2, se(u2, d4)]))
    u2 = filmed_residual_block(embed_cond, u2, 256)
    u2 = residual_block(u2, (64, 64, 256))
    u2 = residual_block(u2, (64, 64, 256))
    u2 = SelfAttention()(u2)

    u3 = residual_block_upscaling(u2, (32, 32, 128))  # 80
    u3 = CoordinateChannel2D()(Concatenate()([u3, se(u3, d3)]))
    u3 = filmed_residual_block(embed_cond, u3, 128)
    u3 = residual_block(u3, (32, 32, 128))
    u3 = residual_block(u3, (32, 32, 128))
    u3 = SelfAttention()(u3)

    s2 = UpSampling2D(4)(Conv2D(1, 1, activation='tanh')(u3))

    u4 = residual_block_upscaling(u3, (16, 16, 64))  # 160
    u4 = CoordinateChannel2D()(Concatenate()([u4, se(u4, d2)]))
    u4 = filmed_residual_block(embed_cond, u4, 64)
    u4 = residual_block(u4, (16, 16, 64))
    u4 = residual_block(u4, (16, 16, 64))
    u4 = SelfAttention()(u4)

    u5 = residual_block_upscaling(u4, (8, 8, 32))  # 320
    u5 = CoordinateChannel2D()(Concatenate()([u5, se(u5, d1)]))
    u5 = filmed_residual_block(embed_cond, u5, 32)
    u5 = residual_block(u5, (8, 8, 32))
    u5 = residual_block(u5, (8, 8, 32))

    u6 = residual_block(u5, (4, 4, 16), True)
    u6 = residual_block(u6, (4, 4, 16))
    u6 = residual_block(u6, (4, 4, 16))

    output_img = Conv2D(1, 1, activation='tanh')(u6)

    return Model([input_cond, input_img], [output_img, s1, s2])


def build_discriminator():
    input_line = Input(shape=IMG_SHAPE)
    input_shade = Input(shape=IMG_SHAPE)
    input_cond = Input((3,))

    input_img = Composite()([input_line, input_shade])

    x = CoordinateChannel2D()(PixelwiseConcat()([input_img, input_cond]))

    x = residual_block_downscaling(x, (8, 8, 32))
    x = residual_block(x, (8, 8, 32))

    x = residual_block_downscaling(x, (16, 16, 64))
    x = residual_block(x, (16, 16, 64))

    x = residual_block_downscaling(x, (32, 32, 128))
    x = residual_block(x, (32, 32, 128))

    x = SelfAttention()(x)

    x = residual_block_downscaling(x, (64, 64, 256))
    x = residual_block(x, (64, 64, 256))

    x = SelfAttention()(x)

    x = residual_block_downscaling(x, (128, 128, 512))
    x = residual_block(x, (128, 128, 512))

    x = GlobalAvgPool2D()(x)
    features = Dropout(0.3)(x)

    x = Dense(256)(features)
    validity = Dense(1, activation='sigmoid')(x)

    return Model([input_cond, input_line, input_shade], validity)
