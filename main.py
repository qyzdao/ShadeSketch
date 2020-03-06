import tensorflow as tf
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Shade Sketches')
parser.add_argument('--image-size', type=int, default=320,
                    help='input image size (default: 320)')
parser.add_argument('--direction', type=str, default='810',
                    help='lighting directions (suggest to choose 810, 210, 710)')
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

if not os.path.exists('norm/'):
    os.makedirs('norm/')

if not os.path.exists('out/'):
    os.makedirs('out/')

# Line norm
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open("linenorm.pb", "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tensors = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        op = sess.graph.get_operations()

        for i, m in enumerate(op):
            print('op{}:'.format(i), m.values())

        inputs = sess.graph.get_tensor_by_name('input_1:0')
        outputs = sess.graph.get_tensor_by_name('conv2d_9/Sigmoid:0')
        s = args.image_size

        for root, dirs, files in os.walk('val/', topdown=False):
            for name in files:
                line_path = os.path.join(root, name)
                print(line_path)

                img = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (s, s))
                img = img.astype(np.float32) / 255.

                img_out = sess.run(outputs, {inputs: np.reshape(img, (1, img.shape[0], img.shape[1], 1))})
                cv2.imwrite(os.path.join('norm/', name), np.squeeze(img_out) * 255.)


# Line shade
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open("lineshader.pb", "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tensors = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        inputs1 = sess.graph.get_tensor_by_name('input_1:0')
        inputs2 = sess.graph.get_tensor_by_name('input_2:0')
        outputs = sess.graph.get_tensor_by_name('conv2d_139/Tanh:0')
        s = args.image_size

        for root, dirs, files in os.walk('norm/', topdown=False):
            for name in files:
                norm_path = os.path.join(root, name)
                print(norm_path)

                img = cv2.imread(norm_path, cv2.IMREAD_GRAYSCALE)
                img = 1 - img.astype(np.float32) / 255. #inverse black-in-white lines to white-in-black

                cond = cond_to_pos(args.direction) # lighting direction
                
                img_out = sess.run(
                    outputs, {
                        inputs1: np.expand_dims(cond, 0),
                        inputs2: np.reshape(img, (1, s, s, 1)),
                    }
                )

                line = cv2.imread(os.path.join('val/', name), cv2.IMREAD_GRAYSCALE)
                line = cv2.resize(line, (s, s))

                shade = (1 - (np.squeeze(img_out) + 1) / 2) * 255. # inverse white-in-black shadow to black-in-white
                final_output = 0.8 * line + 0.2 * shade # composite line drawing and shadow
                cv2.imwrite(os.path.join('out/', name), final_output)

