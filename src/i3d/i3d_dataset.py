#!/usr/bin/env python
#
import itertools
import json
import os
from cv2 import DualTVL1OpticalFlow_create as DualTVL1
from random import shuffle

import cv2
import numpy
from keras.utils import Sequence
from numpy import argmax

from dataset import calc_flow


def resize_new_height(rgb, new_height):
    h, w, c = rgb.shape
    ratio = w / h
    w_resized = int(new_height * ratio)
    resized = cv2.resize(rgb, (w_resized, new_height))
    return resized


def center_crop(rgb, new_height, new_width):
    h, w, c = rgb.shape

    y = int(h / 2 - new_height / 2)
    x = int(w / 2 - new_width / 2)

    return rgb[y:y + new_height, x:x + new_width, :]


class I3DFusionSequence(Sequence):

    def __init__(self, data_dir, train_txt: str, batch_size: int = 32, input_hw=(224, 224), show=False,
                 num_frames: int = 16):
        self.TVL1 = DualTVL1()
        self.input_hw = input_hw
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.all_seq = []
        self.show = show
        self.image_augmentation = None
        # self.image_augmentation = augmentation.ImageAugmentation()

        with open(os.path.join(data_dir, train_txt), 'r') as video_fn:
            train_files = list(map(lambda l: l.strip(), video_fn.readlines()))

        for video_fn in train_files:
            if video_fn.startswith("no_smoking_videos"):
                # hack-hack
                pass
            else:
                human_y = os.path.join(self.data_dir, "jsonl.byhuman", video_fn, 'result.jsonl')
                # machine_y = os.path.join(self.data_dir, "jsonl", _f, 'result.jsonl')

                with open(human_y) as f:
                    predictions_by_frame = [json.loads(s.strip()) for s in f]

                    p_prev = None
                    cls_seq = []
                    for fn_p in predictions_by_frame:
                        fn, p = fn_p
                        cls_id = argmax(p)
                        if p_prev is None:
                            p_prev = p

                        if p_prev[0] == p[0]:
                            cls_seq.append(fn_p)
                        else:
                            cls_seq = []

                        if len(cls_seq) == self.num_frames:
                            self.all_seq.append((cls_id, video_fn, cls_seq))
                            cls_seq = []
                        p_prev = p

        shuffle(self.all_seq)

    def __getitem__(self, index):
        s = index * self.batch_size
        e = min(len(self.all_seq), s + self.batch_size)

        xrgb_batch = numpy.zeros(shape=(self.batch_size, self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        xflow_batch = numpy.zeros(shape=(self.batch_size, self.num_frames, self.input_hw[0], self.input_hw[1], 2))
        ybatch = numpy.zeros(shape=(self.batch_size, 2))
        for ii in range(s, e):
            i = ii - s
            xrgb, xflow, y = self.get_one_xy(ii)
            xrgb_batch[i, :, :, :, :] = xrgb
            xflow_batch[i, :, :, :, :] = xflow
            ybatch[i, :] = y

        return (xrgb_batch, xflow_batch), ybatch

    def get_one_xy(self, index):
        if self.image_augmentation:
            self.image_augmentation.renew()  # change random sequence for imgaug

        v = self.all_seq[index]
        cls_id, v_file, cls_seq = v
        IDX_FRAMENUMBER = 0
        start_frame = cls_seq[0][IDX_FRAMENUMBER]
        end_frame = start_frame + len(cls_seq)

        video_path = os.path.join(self.data_dir, v_file)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fn_counter = itertools.count()
        xframes = numpy.zeros(shape=(self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        xflow = numpy.zeros(shape=(self.num_frames, self.input_hw[0], self.input_hw[1], 2))

        prev_gray = None
        while cap.isOpened():
            ret, bgr = cap.read()

            fn = next(fn_counter)
            if bgr is None or fn == self.num_frames:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = resize_new_height(rgb, self.input_hw[0])
            rgb = center_crop(rgb, self.input_hw[0], self.input_hw[1])
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            if prev_gray is None:
                prev_gray = gray

            if self.image_augmentation:
                rgb = self.image_augmentation.blur.augment_images([rgb])[0]
                rgb = self.image_augmentation.transform.augment_images([rgb])[0]

            cur_flow = calc_flow(gray, prev_gray)
            xflow[fn, :, :, :] = cur_flow
            prev_gray = gray

            if self.show:
                print(video_path)
                cv2.imshow("%d f%d" % (fn + start_frame, cls_id), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(25)

            rgb = rgb / 127.5 - 1
            xframes[fn, :, :, :] = rgb

        # Crop to 112x112
        # reshape_frames = xframes[:, 8:120, 30:142, :]

        # # Resize to 112x112
        # reshape_frames = numpy.zeros(shape=(self.num_frames, 112, 112, 3))
        # for i in range(xframes.shape[0]):
        #     reshape_frames[i, :, :, :] = cv2.resize(xframes[i, :, :, :], (112, 112))

        y = numpy.zeros(shape=(1, 2))
        y[0][cls_id] = 1.
        return xframes, xflow, y

    def __len__(self):
        return int(numpy.ceil(len(self.all_seq) / self.batch_size))


if __name__ == '__main__':
    seq = I3DFusionSequence("/Volumes/bstorage/datasets/vg_smoke/", "train.txt",
                            input_hw=(224, 224), batch_size=32, num_frames=8,
                            show=True)
    print(len(seq))
    # val = C3DSequence("/Volumes/bstorage/datasets/vg_smoke/", "validate.txt")

    for i in range(len(seq)):
        print(seq[i])
