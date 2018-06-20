#!/usr/bin/env python
import argparse
import itertools
import json

import cv2
import numpy
from keras.models import load_model

from i3d_dataset import center_crop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf', type=str, help="path to model h5 file")
    parser.add_argument('video_file', type=str, help="path to the video file")
    parser.add_argument('out', type=str, help="path to jsonl output")
    args = parser.parse_args()

    m = load_model(args.hdf)
    input_shape = m.input.shape[1:]
    nframes = int(input_shape[0])
    x = numpy.zeros(shape=input_shape)

    print(input_shape)
    with open(args.out, 'w') as _f:
        cap = cv2.VideoCapture(args.video_file)

        fn_iter = itertools.count()
        i = 0
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                cap.release()
                break

            fn = next(fn_iter)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = center_crop(rgb, int(input_shape[1]), int(input_shape[2]))
            rgb = rgb / 127.5 - 1
            x[i, :, :, :] = rgb

            if i == nframes - 1:
                for y in m.predict(numpy.stack([x, ])):
                    for _fn in range(fn + 1 - nframes, fn):
                        fn_y = [_fn, y.round(2).tolist()]
                        s = json.dumps(fn_y)
                        _f.write("%s\n" % s)
                        print(s)

                i = 0
                x[:, :, :, :] = 0
            i = i + 1
