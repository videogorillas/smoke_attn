#!/usr/bin/env python
#
import csv
import itertools
import json
import os
from random import shuffle, randint

import cv2
import numpy
from keras.utils import Sequence
from numpy import argmax

from dataset import calc_flow


def scale_to_new_height(rgb, new_height):
    h, w, c = rgb.shape
    ratio = w / h
    w_resized = int(new_height * ratio)
    resized = cv2.resize(rgb, (w_resized, new_height))
    return resized


def crop_xy(rgb, new_height, new_width, x, y):
    h, w, c = rgb.shape
    x = min(x, w - new_width)
    y = min(y, h - new_height)
    return rgb[y:y + new_height, x:x + new_width, :]


def center_crop(rgb, new_height, new_width, jitter=True):
    h, w, c = rgb.shape

    y = int(h / 2 - new_height / 2)
    x = int(w / 2 - new_width / 2)

    # jitter
    if jitter:
        x = randint(0, x)
        y = randint(0, y)

    return rgb[y:y + new_height, x:x + new_width, :]


def load_activity_net_positives(data_dir):
    with open(data_dir + '/activity_net-positives.csv') as _f:
        rdr = csv.reader(_f)

        for row in rdr:
            _, id, start_sec, end_sec, _ = row
            print(id, start_sec, end_sec)

            cap = cv2.VideoCapture(data_dir + "/activity_net/%s.mp4" % id)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_POS_FRAMES)

            s_fn = int(fps * float(start_sec))
            e_fn = int(fps * float(end_sec))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s_fn))

            fn_iter = itertools.count(s_fn)
            while cap.isOpened():
                ret, bgr = cap.read()
                fn = next(fn_iter)
                if ret and fn < e_fn:
                    cv2.imshow("a", bgr)
                    cv2.waitKey(25)
                else:
                    break

            cap.release()


def load_sequences(data_dir, num_frames_in_sequence, train_txt):
    all_seq = []
    cls1_count = 0
    cls0_count = 0
    if train_txt == "train.txt":
        print("Loading activity_net", train_txt)
        # Load activity-net positives {{{
        with open(data_dir + '/activity_net-positives.csv') as _f:
            rdr = csv.reader(_f)

            cap = cv2.VideoCapture()
            for row in rdr:
                _, id, start_sec, end_sec, _ = row

                video_fn = "activity_net/%s.mp4" % id
                # cap = cv2.VideoCapture(data_dir + "/" + video_fn)
                cap.open(data_dir + "/" + video_fn)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                _start_fn = int(fps * float(start_sec))
                _end_fn = int(fps * float(end_sec))
                # print(id, _start_fn, _end_fn)

                for moment_start in range(_start_fn,
                                          min(_end_fn + num_frames_in_sequence + 4, total_frames),
                                          num_frames_in_sequence):
                    start_fn = moment_start
                    end_fn = start_fn + num_frames_in_sequence

                    # train sequence
                    cls_seq = []
                    for _fn in range(start_fn, end_fn):
                        cls_seq.append((_fn, [0, 1.0]))
                    all_seq.append((1, video_fn, cls_seq))
                    cls1_count = cls1_count + 1
        # }}}
    with open(os.path.join(data_dir, train_txt), 'r') as video_fn:
        train_files = list(map(lambda l: l.strip(), video_fn.readlines()))
    for video_fn in train_files:
        if video_fn.startswith("no_smoking_videos"):
            # Folder for negatives

            # Choose random clip from the video
            cap = cv2.VideoCapture(os.path.join(data_dir, video_fn))
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            for moment_start in range(0, int(total_frames / num_frames_in_sequence) - 1):
                start_fn = moment_start
                end_fn = start_fn + num_frames_in_sequence

                # train sequence
                cls_seq = []
                for _fn in range(start_fn, end_fn):
                    cls_seq.append((_fn, [1.0, 0]))
                all_seq.append((0, video_fn, cls_seq))
                cls0_count = cls0_count + 1
            continue

        human_y = os.path.join(data_dir, "jsonl.byhuman", video_fn, 'result.jsonl')
        # machine_y = os.path.join(self.data_dir, "jsonl", _f, 'result.jsonl')

        with open(human_y) as f:
            predictions_by_frame = [json.loads(s.strip()) for s in f]

            p_prev = None
            cls_seq = []
            for fn_p in predictions_by_frame:
                fn, p = fn_p
                cls_id = int(argmax(p))
                if p_prev is None:
                    p_prev = p

                if p_prev[0] == p[0]:
                    cls_seq.append(fn_p)

                if len(cls_seq) == num_frames_in_sequence:
                    all_seq.append((cls_id, video_fn, cls_seq))

                    # Stats
                    if cls_id == 1:
                        cls1_count = cls1_count + 1
                    if cls_id == 0:
                        cls0_count = cls0_count + 1

                    cls_seq = []
                p_prev = p

    print("Class distribution: cls1=%d; cls0=%d" % (cls1_count, cls0_count))
    return all_seq


class I3DFusionSequence(Sequence):

    def __init__(self, data_dir, train_txt: str, batch_size: int = 32, input_hw=(224, 224), show=False,
                 num_frames_in_sequence: int = 16):
        self.input_hw = input_hw
        self.num_frames = num_frames_in_sequence
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.all_seq = []
        self.show = show
        self.image_augmentation = None
        # self.image_augmentation = augmentation.ImageAugmentation()

        self.cap = cv2.VideoCapture()

        all_seq_fn = train_txt + "_all_seq.json"
        if os.path.isfile(all_seq_fn):
            with open(all_seq_fn, 'r') as _f:
                self.all_seq = json.load(_f)
        else:
            self.all_seq = load_sequences(data_dir, num_frames_in_sequence, train_txt)
            with open(all_seq_fn, 'w') as _f:
                json.dump(self.all_seq, _f)

        assert len(self.all_seq) > 0, "empty dataset. something is wrong"

        # kinda this: all_seq = [  (1, video_fn, cls_seq), ..., ...  ]
        shuffle(self.all_seq)

    def __getitem__(self, index):
        s = index * self.batch_size
        e = min(len(self.all_seq), s + self.batch_size)

        xrgb_batch = numpy.zeros(shape=(self.batch_size, self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        xflow_batch = numpy.zeros(shape=(self.batch_size, self.num_frames, self.input_hw[0], self.input_hw[1], 2))
        ybatch = numpy.zeros(shape=(self.batch_size, 2))

        for ii in range(s, e):
            i = ii - s
            xrgb, xflow, y = self.get_one_xy(ii, cap=self.cap)
            xrgb_batch[i, :, :, :, :] = xrgb
            xflow_batch[i, :, :, :, :] = xflow
            ybatch[i, :] = y

        return [xrgb_batch, xflow_batch], ybatch

    def get_one_xy(self, index, cap=cv2.VideoCapture()):
        if self.image_augmentation:
            self.image_augmentation.renew()  # change random sequence for imgaug

        v = self.all_seq[index]
        cls_id, v_file, cls_seq = v
        IDX_FRAMENUMBER = 0
        start_frame = cls_seq[0][IDX_FRAMENUMBER]
        end_frame = start_frame + len(cls_seq)

        video_path = os.path.join(self.data_dir, v_file)
        # cap = cv2.VideoCapture(video_path)
        cap.open(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fn_counter = itertools.count()
        xframes = numpy.zeros(shape=(self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        xflow = numpy.zeros(shape=(self.num_frames, self.input_hw[0], self.input_hw[1], 2))

        prev_gray = None
        x = randint(0, 32)
        y = randint(0, 32)
        while cap.isOpened():
            ret, bgr = cap.read()

            fn = next(fn_counter)
            if bgr is None or fn == self.num_frames:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = scale_to_new_height(rgb, self.input_hw[0])
            # rgb = center_crop(rgb, self.input_hw[0], self.input_hw[1], jitter=False)
            rgb = crop_xy(rgb, self.input_hw[0], self.input_hw[1], x, y)
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
                # cv2.imshow("%d f%d" % (fn + start_frame, cls_id), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow("clsid%d" % (cls_id), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(25)

            rgb = rgb / 127.5 - 1
            xframes[fn, :, :, :] = rgb

        cap.release()
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
                            input_hw=(224, 224), batch_size=32, num_frames_in_sequence=32,
                            show=True)
    print(len(seq))
    # val = C3DSequence("/Volumes/bstorage/datasets/vg_smoke/", "validate.txt")

    for i in range(len(seq)):
        print(seq[i])
