#!/usr/bin/env python
import csv
import itertools
import os
from random import randint

import cv2
import image2pipe as image2pipe
import numpy
from keras.utils import Sequence


class BFSequence(Sequence):

    def __init__(self, data_dir, train_csv: str,
                 batch_size: int = 32, input_hw=(224, 224), show=False,
                 num_frames_in_sequence: int = 16, only_temporal=False, only_spacial=False):
        self.only_spacial = only_spacial
        self.only_temporal = only_temporal
        self.input_hw = input_hw
        self.num_frames = num_frames_in_sequence
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.show = show
        self.data = []

        with open(train_csv) as _f:
            csvrdr = csv.reader(_f)

            for row in csvrdr:
                movieId1, start1, end1, movieId2, start2, end2 = row
                self.data.append((movieId1, int(start1), int(end1),
                                  movieId2, int(start2), int(end2)))

    def __len__(self):
        return int(numpy.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        s = index * self.batch_size
        e = min(len(self.data), s + self.batch_size)

        xrgb1_batch = numpy.zeros(shape=(self.batch_size, self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        xrgb2_batch = numpy.zeros(shape=(self.batch_size, self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        ybatch = numpy.zeros(shape=(self.batch_size, self.num_frames))

        for datum_index in range(s, e):
            i = datum_index - s
            x1, x2, y = self.get_one_xy(index=datum_index)
            xrgb1_batch[i, :, :, :, :] = x1
            xrgb2_batch[i, :, :, :, :] = x2
            ybatch[i, :] = y

            return [xrgb1_batch, xrgb2_batch], ybatch

    def get_one_xy(self, index):
        datum = self.data[index]
        movieId1, start1, end1, movieId2, start2, end2 = datum
        if self.show:
            print(index, datum)

        v_file1 = os.path.join(self.data_dir, movieId1)
        video_path1 = os.path.join(self.data_dir, v_file1)
        cap1 = cv2.VideoCapture()
        cap1.open(video_path1)
        fps1 = cap1.get(cv2.CAP_PROP_FPS)

        """
        start1, end1
        start2, end2
        [1, 1, 1]

        shift=1
        [0, 1, 1]
        """

        # the sequence may be longer than we need. so start every time from a random frame
        # offset = randint(0, end1 - start1 - self.num_frames - 1)
        offset = randint(-15, 15)
        max(0, start1)

        # shift = randint(-1 * (max(0, start2 - self.num_frames)), self.num_frames - 1)

        # seek_to_frame(cap1, start1 + offset)
        seek_to_frame(cap1, start1, fps1)

        fn_counter = itertools.count()
        x1 = numpy.zeros(shape=(self.num_frames, self.input_hw[0], self.input_hw[1], 3))

        while cap1.isOpened():
            ret, bgr = cap1.read()
            fn = next(fn_counter)
            if bgr is None or fn == self.num_frames:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, self.input_hw)

            x1[fn, :, :, :] = rgb

        cap1.release()
        del cap1

        v_file2 = os.path.join(self.data_dir, movieId2)
        video_path2 = os.path.join(self.data_dir, v_file2)
        cap2 = cv2.VideoCapture()
        cap2.open(video_path2)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)

        shift = randint(-5, 5)
        # seek_to_frame(cap2, start2 + shift)
        seek_to_frame(cap2, start2, fps2)
        x2 = numpy.zeros(shape=(self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        fn_counter = itertools.count()
        while cap2.isOpened():
            ret, bgr = cap2.read()
            fn = next(fn_counter)
            if bgr is None or fn == self.num_frames:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, self.input_hw)
            x2[fn, :, :, :] = rgb

        cap2.release()
        del cap2

        y = numpy.zeros(shape=(1, self.num_frames))
        print("shift:", shift)
        if shift >= 0:
            y[0][shift:] = 1.
        else:
            y[0][:shift] = 1.

        if self.show:
            for i in range(0, self.num_frames):
                showtwo(cv2.cvtColor(x1[i].astype(numpy.uint8), cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(x2[i].astype(numpy.uint8), cv2.COLOR_RGB2BGR))
                cv2.waitKey(25)

            cv2.waitKey(0)
        x1 = x1 / 127.5 - 1
        x2 = x2 / 127.5 - 1
        return x1, x2, y


def seek_to_frame(cap, fn, fps):
    msec = 1000.0 * fn / fps
    print("seek to fn ", fn, msec, fps)
    print(cap.set(cv2.CAP_PROP_POS_MSEC, msec))


def showtwo(bgr1, bgr2):
    vstack = numpy.vstack([bgr1, bgr2])
    vstack = cv2.resize(vstack, (400, 800))
    cv2.imshow('two', vstack)


def seek_test():
    cap1 = cv2.VideoCapture("/blender/storage/home/chexov/macg/MACG_S02_Ep024_ING_5764188.mov")
    cap2 = cv2.VideoCapture("/blender/storage/home/chexov/macg/BX137_SRNA_07.mov")

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    # 256 ('MACG_S02_Ep024_ING_5764188.mov', 48952, 48952, 'BX137_SRNA_07.mov', 330, 330)
    # 288 ('MACG_S02_Ep024_ING_5764188.mov', 49008, 49008, 'BX137_SRNA_07.mov', 443, 443)

    seek_to_frame(cap1, 49041, fps1)
    seek_to_frame(cap2, 508, fps2)
    ret, bgr1 = cap1.read()
    ret, bgr2 = cap2.read()
    showtwo(bgr1, bgr2)
    cv2.waitKey(0)

    seek_to_frame(cap1, 48952, fps1)
    seek_to_frame(cap2, 330, fps2)
    ret, bgr1 = cap1.read()
    ret, bgr2 = cap2.read()
    showtwo(bgr1, bgr2)
    cv2.waitKey(0)

    seek_to_frame(cap1, 49008, fps1)
    seek_to_frame(cap2, 443, fps2)
    ret, bgr1 = cap1.read()
    ret, bgr2 = cap2.read()
    showtwo(bgr1, bgr2)
    cv2.waitKey(0)

    seek_to_frame(cap1, 49041, fps1)
    seek_to_frame(cap2, 508, fps2)
    ret, bgr1 = cap1.read()
    ret, bgr2 = cap2.read()
    showtwo(bgr1, bgr2)
    cv2.waitKey(0)


if __name__ == '__main__':
    # all_samples = load_dataset_sequences("/Volumes/bstorage/datasets/vg_smoke/", "train.txt")
    # 
    # print("Sample (cls_id, video_fn, start_frame, end_frame)", all_samples[0])
    # print("Total samples: %d" % len(all_samples))
    # 
    # pos = filter(lambda x: x[0] == 1.0, all_samples)
    # neg = filter(lambda x: x[0] == 0, all_samples)
    # print("Positives seq: %d" % len(list(pos)))
    # print("Neg seq: %d" % len(list(neg)))
    # 
    # good_samples = filter(lambda x: x[3] - x[2] > 32, all_samples)
    # print("Good samples: %d" % len(list(good_samples)))
    # 
    # h = sorted(map(lambda x: x[3] - x[2], all_samples))
    # mean = numpy.mean(h)
    # std = numpy.std(h)
    # print("mean: ", mean)
    # print(" std: ", std)
    # print(" max: ", numpy.max(h))
    # print(" min: ", numpy.min(h))
    # fit = norm.pdf(h, mean, std)
    # 
    # pyplot.plot(h, fit, '-o')
    # pyplot.hist(h, normed=1)
    # pyplot.show()

    seq = BFSequence("/blender/storage/home/chexov/macg/", "truth.csv",
                     input_hw=(524, 524), batch_size=32, num_frames_in_sequence=32,
                     show=True)
    print("total batches with samples", len(seq))
    # val = C3DSequence("/Volumes/bstorage/datasets/vg_smoke/", "validate.txt")

    # image2pipe.images_from_url('')
    seek_test()
    # 
    for i in range(len(seq)):
        print(seq[i][1][0])
