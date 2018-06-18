'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import argparse
import json
from itertools import count

import cv2
import numpy as np
from keras.models import load_model

from dataset import calc_flow

NUM_FRAMES = 32
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400


def m2(model, video_f, show=False):
    inputs = []
    flow = []

    cap = cv2.VideoCapture(video_f)
    prev = None
    fn_counter = count()
    fn_startrange = 0
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

        curr_flow = calc_flow(gray, prev)
        flow.append(curr_flow)
        prev = gray
        if show:
            cv2.imshow("cropped", cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))

        rgb = rgb / 127.5 - 1
        inputs.append(rgb)

        if len(inputs) == NUM_FRAMES:
            x_rgb = np.array([inputs])
            x_flow = np.array([flow])
            y_batch = model.predict([x_rgb, x_flow])

            y = y_batch[0]
            for i in range(fn_startrange, fn):
                yield [i, y.tolist()]

            if show:
                cls_id = np.argmax(y)
                cv2.imshow("%s" % cls_id, resized)
                cv2.waitKey(25)

            inputs = []
            flow = []
            fn_startrange = fn


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf', type=str, help="path to model h5 file")
    parser.add_argument('video_file', type=str, help="path to the video file")
    parser.add_argument('out', type=str, help="path to jsonl output")
    args = parser.parse_args()

    m = load_model(args.hdf)
    with open(args.out, 'w') as _f:
        for fn_y in m2(m, args.video_file, show=False):
            s = json.dumps(fn_y)
            print(s)
            _f.write("%s\n" % s)
