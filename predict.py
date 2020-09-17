"""
ShadeSketch
https://github.com/qyzdao/ShadeSketch

Learning to Shadow Hand-drawn Sketches
Qingyuan Zheng, Zhuoru Li, Adam W. Bargteil

Copyright (C) 2020 The respective authors and Project HAT. All rights reserved.
Licensed under MIT license.
"""

import tensorflow

if hasattr(tensorflow.compat, 'v1'):
    tf = tensorflow.compat.v1
    tf.disable_v2_behavior()
else:
    tf = tensorflow

import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='ShadeSketch')
parser.add_argument('--input-dir', type=str, default='./val', help='input directory')
parser.add_argument('--output-dir', type=str, default='./output', help='output directory')
parser.add_argument('--image-size', type=int, default=320, help='input image size (default: 320)')
parser.add_argument('--direction', type=str, default='810', help='light direction (suggest to choose 810, 210, 710)')
parser.add_argument('--threshold', type=int, default=200, help='threshold value, 0 disable (default: 200)')
parser.add_argument('--use-smooth', action="store_true", default=False, help='use smooth')
parser.add_argument('--use-norm', action="store_true", default=False, help='use norm')
args = parser.parse_args()


def cond_to_pos(cond):
    cond_pos_rel = {
        '002': [0, 0, -1],
        '110': [0, 1, -1], '210': [1, 1, -1], '310': [1, 0, -1], '410': [1, -1, -1], '510': [0, -1, -1],
        '610': [-1, -1, -1], '710': [-1, 0, -1], '810': [-1, 1, -1],
        '120': [0, 1, 0], '220': [1, 1, 0], '320': [1, 0, 0], '420': [1, -1, 0], '520': [0, -1, 0], '620': [-1, -1, 0],
        '720': [-1, 0, 0], '820': [-1, 1, 0],
        '130': [0, 1, 1], '230': [1, 1, 1], '330': [1, 0, 1], '430': [1, -1, 1], '530': [0, -1, 1], '630': [-1, -1, 1],
        '730': [-1, 0, 1], '830': [-1, 1, 1],
        '001': [0, 0, 1]
    }
    return cond_pos_rel[cond]


def normalize_cond(cond_str):
    _cond_str = cond_str.strip()

    if len(_cond_str) == 3:
        return cond_to_pos(_cond_str)

    if ',' in _cond_str:
        raw_cond = _cond_str.replace('[', '').replace(']', '').split(',')
        if len(raw_cond) == 3:
            return raw_cond

    return [-1, 1, -1]


def predict():
    output_dir = args.output_dir
    input_dir = args.input_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load models
    with tf.gfile.FastGFile('./models/linesmoother.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='lineSmoother')

    with tf.gfile.FastGFile('./models/linenorm.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='lineNorm')

    with tf.gfile.FastGFile('./models/lineshader.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='lineShader')

    # Run through folders
    with tf.Session() as sess:
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for name in files:
                line_path = os.path.join(root, name)
                print('Running inference for %s ...' % line_path)

                img = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)

                # Resize image
                s = args.image_size
                h, w = img.shape[:2]

                imgrs = cv2.resize(img, (s, s))

                # Threshold image
                if args.threshold > 0:
                    _, imgrs = cv2.threshold(imgrs, args.threshold, 255, cv2.THRESH_BINARY)

                # Prepare for inference
                tensors = np.reshape(imgrs, (1, s, s, 1)).astype(np.float32) / 255.
                ctensors = np.expand_dims(normalize_cond(args.direction), 0)

                # Run inference
                if args.use_smooth or args.threshold > 0:
                    tensors = sess.run(
                        'lineSmoother/conv2d_9/Sigmoid:0',
                        {
                            'lineSmoother/input_1:0': tensors
                        }
                    )
                    smoothResult = tensors

                if args.use_norm:
                    tensors = sess.run(
                        'lineNorm/conv2d_9/Sigmoid:0',
                        {
                            'lineNorm/input_1:0': tensors
                        }
                    )
                    normResult = tensors

                tensors = sess.run(
                    'lineShader/conv2d_139/Tanh:0',
                    {
                        'lineShader/input_1:0': ctensors,
                        'lineShader/input_2:0': 1. - tensors
                    }
                )
                shadeResult = tensors

                # Save result
                shade = (1 - (np.squeeze(shadeResult) + 1) / 2) * 255.
                shade = cv2.resize(shade, (w, h))

                comp = 0.8 * img + 0.2 * shade

                cv2.imwrite(os.path.join('./output', name), comp)


if __name__ == '__main__':
    predict()
