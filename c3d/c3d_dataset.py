#!/usr/bin/env python
#
import itertools
import json
import os
from random import shuffle

import cv2
import numpy
from keras.utils import Sequence, get_file
from numpy import argmax

import augmentation
from sports1M_utils import C3D_MEAN_PATH


class C3DSequence(Sequence):

    def __init__(self, data_dir, train_txt: str, batch_size: int = 32, show=False):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.all_seq = []
        self.show = show
        self.image_augmentation = augmentation.ImageAugmentation()

        mean_path = get_file('c3d_mean.npy',
                             C3D_MEAN_PATH,
                             cache_subdir='models',
                             md5_hash='08a07d9761e76097985124d9e8b2fe34')
        # Subtract mean
        self.mean = numpy.load(mean_path)

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

                        if len(cls_seq) == 16:
                            self.all_seq.append((cls_id, video_fn, cls_seq))

                            cls_seq = []
                        p_prev = p

        shuffle(self.all_seq)

    def __getitem__(self, index):

        s = index * self.batch_size
        e = min(len(self.all_seq), s + self.batch_size)

        xbatch = numpy.zeros(shape=(self.batch_size, 16, 112, 112, 3))
        ybatch = numpy.zeros(shape=(self.batch_size, 2))
        for i in range(self.batch_size):
            x, y = self.get_one_xy(i)
            xbatch[i, :, :, :, :] = x
            ybatch[i, :] = y

        return xbatch, ybatch

    def get_one_xy(self, index):
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
        xframes = numpy.zeros(shape=(16, 128, 171, 3))
        while cap.isOpened():
            ret, bgr = cap.read()

            fn = next(fn_counter)
            if bgr is None or fn == 16:
                break

            bgr = cv2.resize(bgr, (171, 128))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = self.image_augmentation.blur.augment_images([rgb])[0]
            rgb = self.image_augmentation.transform.augment_images([rgb])[0]

            xframes[fn, :, :, :] = rgb
            xframes -= self.mean

            if self.show:
                print(video_path)
                cv2.imshow("f%d" % cls_id, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(25)

        # Crop to 112x112
        # reshape_frames = xframes[:, 8:120, 30:142, :]

        # Resize to 112x112
        reshape_frames = numpy.zeros(shape=(16, 112, 112, 3))
        for i in range(xframes.shape[0]):
            reshape_frames[i, :, :, :] = cv2.resize(xframes[i, :, :, :], (112, 112))

        y = numpy.zeros(shape=(1, 2))
        y[0][cls_id] = 1.
        return reshape_frames, y

    def __len__(self):
        return int(numpy.ceil(len(self.all_seq) / self.batch_size))


if __name__ == '__main__':
    seq = C3DSequence("/Volumes/bstorage/datasets/vg_smoke/", "train.txt", batch_size=32, show=True)
    # val = C3DSequence("/Volumes/bstorage/datasets/vg_smoke/", "validate.txt")

    for i in range(len(seq)):
        print(seq[i])
