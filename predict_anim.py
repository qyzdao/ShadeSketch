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
import glob

parser = argparse.ArgumentParser(description='ShadeSketch')
parser.add_argument('--input', type=str, default='./val/1.png', help='image name to make gif')
parser.add_argument('--output-dir', type=str, default='./output', help='output directory')
parser.add_argument('--image-size', type=int, default=320, help='input image size (default: 320)')
parser.add_argument('--light-depth', type=str, default='front', help='light depth (front, side)')
parser.add_argument('--threshold', type=int, default=200, help='threshold value, 0 disable (default: 200)')
parser.add_argument('--use-smooth', action="store_true", default=False, help='use smooth')
parser.add_argument('--use-norm', action="store_true", default=False, help='use norm')
parser.add_argument('--pack-gif', action="store_true", default=False, help='pack GIF, do not works on Windows')
args = parser.parse_args()


def predict_gif():
    input_dir = args.input
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input).split('.')[0])

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

    print('Running inference for %s ...' % input_dir)
    img = cv2.imread(input_dir, cv2.IMREAD_GRAYSCALE)

    # Resize image
    s = args.image_size
    h, w = img.shape[:2]

    imgrs = cv2.resize(img, (s, s))

    # Threshold image
    if args.threshold > 0:
        _, imgrs = cv2.threshold(imgrs, args.threshold, 255, cv2.THRESH_BINARY)

    # Prepare for inference
    tensors = np.reshape(imgrs, (1, s, s, 1)).astype(np.float32) / 255.

    with tf.Session() as sess:
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

        # 'cond' is lighting direction.
        # The default is front lighting. (810-210, 210-410...)
        if args.light_depth == 'front':
            # front lighting
            conds = [
                # 810-210
                [-1, 1, -1],
                # 210-410
                [1, 1, -1],
                # 410-610
                [1, -1, -1],
                # 610-810
                [-1, -1, -1]
            ]
        elif args.light_depth == 'side':
            # side lighting
            conds = [
                # 820-220
                [-1, 1, 0],
                # 220-420
                [1, 1, 0],
                # 420-620
                [1, -1, 0],
                # 620-820
                [-1, -1, 0]
            ]
        else:
            conds = []

        for j, cond in enumerate(conds):
            for i in range(20):
                shadeResult = sess.run(
                    'lineShader/conv2d_139/Tanh:0',
                    {
                        'lineShader/input_1:0': np.expand_dims(cond, 0),
                        'lineShader/input_2:0': 1 - tensors
                    }
                )
                shade = (1 - (np.squeeze(shadeResult) + 1) / 2) * 255.
                shade = cv2.resize(shade, (w, h))

                comp = 0.8 * img + 0.2 * shade

                cv2.imwrite(os.path.join(output_dir, '%s.png' % str(i + j * 20)), comp)

                if j == 0:
                    cond[0] = cond[0] + 0.1
                elif j == 1:
                    cond[1] = cond[1] - 0.1
                elif j == 2:
                    cond[0] = cond[0] - 0.1
                elif j == 3:
                    cond[1] = cond[1] + 0.1


def make_gif():
    print('Packing GIF, only works on Mac.')

    input_name = os.path.basename(args.input).split('.')[0]
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input).split('.')[0])
    file_list = glob.glob(os.path.join(output_dir, '*.png'))

    list.sort(file_list, key=lambda x: int(os.path.basename(x).split('.')[0]))

    with open('./image_list.txt', 'w') as file:
        for item in file_list:
            file.write("%s\n" % os.path.abspath(item))

    # On windows convert is 'magick'
    try:
        os.system('convert @image_list.txt {}.gif'.format(input_name))
    except:
        print('Convert GIF failed. Convert only works on Mac, you can use other tools for GIF.')


if __name__ == '__main__':
    predict_gif()

    if args.pack_gif:
        make_gif()
