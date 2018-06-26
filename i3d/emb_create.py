'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import argparse
from itertools import count

import cv2
import numpy
import numpy as np
from keras import Input, Model

from i3d_inception import Inception_Inflated3d

NUM_FRAMES = 32
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400


def main_fe(model, video_f, show=False):
    inputs = []

    cap = cv2.VideoCapture(video_f)
    prev = None
    fn_counter = count()
    while cap.isOpened():
        ret, bgr = cap.read()
        fn = next(fn_counter)

        if bgr is None:
            break

        h, w, c = bgr.shape
        ratio = w / h
        w_resized = int(224 * ratio)
        resized = cv2.resize(bgr, (w_resized, 224))
        # cv2.imshow("resized", resized)
        # cv2.waitKey(0)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        w1 = int((w_resized / 2) - 224 / 2)
        rgb = rgb[0:224, w1:w1 + 224, :]

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if prev is None:
            prev = gray

        # curr_flow = calc_flow(gray, prev)
        # flow.append(curr_flow)
        prev = gray
        if show:
            cv2.imshow("cropped", cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))

        rgb = rgb / 127.5 - 1
        inputs.append(rgb)

        if fn + 1 < NUM_FRAMES:
            continue

        x_rgb = np.array([inputs])
        y_vec = model.predict(x_rgb)
        yield y_vec

        inputs = inputs[1:]


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file', type=str, help="path to the video file")
    parser.add_argument('outvecs', type=str, help="path to the opeput numpy vector")
    args = parser.parse_args()

    rgb_input = Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS))
    fe = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        # weights='rgb_kinetics_only',
        input_tensor=rgb_input,
        classes=-1)

    m = Model(input=fe.get_input_at(0), output=(fe.get_layer("Mixed_5c").output))
    # plot_model(m, show_shapes=True)

    vecs = []
    for vec in main_fe(m, args.video_file, show=False):
        print(vec.sum())
        vecs.append(vec.flatten())

    numpy.save(args.outvecs, numpy.array(vecs))
