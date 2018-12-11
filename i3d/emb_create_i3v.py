'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import argparse
import os
from itertools import count

import cv2
import numpy
from keras import Input, Model
from keras.applications import InceptionV3

NUM_FRAMES = 32
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400


def main_fe(model, video_f, show=False):
    cap = cv2.VideoCapture(video_f)
    fn_counter = count()
    while cap.isOpened():
        ret, bgr = cap.read()
        fn = next(fn_counter)

        if bgr is None:
            break

        resized = cv2.resize(bgr, (299, 299))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb / 127.5 - 1

        if fn + 1 < NUM_FRAMES:
            continue

        x = numpy.stack([rgb, ])
        for y_vec in model.predict(x):
            yield fn, y_vec
    cap.release()
    del cap


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file', type=str, help="path to the video file")
    parser.add_argument('outvecs', type=str, help="path to the opeput numpy vector")
    args = parser.parse_args()

    rgb_input = Input(shape=(299, 299, 3))
    fe = InceptionV3(include_top=False, weights="imagenet", input_tensor=rgb_input)
    # plot_model(fe, show_shapes=True)

    # m = Model(input=fe.get_input_at(0), output=(fe.get_layer("mixed9").output))
    m = fe

    os.makedirs(args.outvecs, exist_ok=True)

    for fn, vec in main_fe(m, args.video_file, show=False):
        print(fn)
        t = vec.flatten()
        numpy.save("%s/%09d.npy" % (args.outvecs, fn), t)
