#!/usr/bin/env python
#
import csv
import functools
import itertools
import json
import os
from random import shuffle, randint

import cv2
import numpy
from keras.utils import Sequence
from numpy import argmax

from dataset import calc_flow


def jsonl_to_sequences(predictions_by_frame):
    result = []
    cls_id_prev = -1
    cls_seq = []
    for fn_y in predictions_by_frame:
        fn, y = fn_y
        cls_id = int(argmax(y))

        if cls_id_prev == -1:
            cls_id_prev = cls_id

        if cls_id_prev == cls_id:
            cls_seq.append(fn)
        else:
            result.append((cls_id_prev, cls_seq))
            cls_seq = [fn, ]
        cls_id_prev = cls_id
    return result


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


def load_dataset_sequences(data_dir, train_txt):
    dataset_sequences = []
    if train_txt == "train.txt":
        print("Loading activity_net subset", train_txt)
        load_activity_net_seq(data_dir, dataset_sequences)

        print("Loading kinetics600 subset", train_txt)
        load_kinetics600("/blender/storage/datasets/kinetics-600/",
                         data_dir + '/k600.csv', dataset_sequences)

    with open(os.path.join(data_dir, train_txt), 'r') as video_fn:
        train_files = list(map(lambda l: l.strip(), video_fn.readlines()))

    for video_fn in train_files:
        if video_fn.startswith("no_smoking_videos"):
            # Folder for negatives

            # Choose random clip from the video
            cap = cv2.VideoCapture(os.path.join(data_dir, video_fn))
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            if total_frames > 1:
                dataset_sequences.append((0, video_fn, 0, total_frames - 1))
            continue

        human_y = os.path.join(data_dir, "jsonl.byhuman", video_fn, 'result.jsonl')
        # machine_y = os.path.join(self.data_dir, "jsonl", _f, 'result.jsonl')

        with open(human_y) as f:
            predictions_by_frame = [json.loads(s.strip()) for s in f]
            for _seq in jsonl_to_sequences(predictions_by_frame):
                cls_id, fnumbers = _seq

                _start_fn = fnumbers[0]
                _end_fn = fnumbers[-1]
                if _end_fn > _start_fn:
                    dataset_sequences.append((cls_id, video_fn, _start_fn, _end_fn))

    return dataset_sequences


def load_activity_net_seq(data_dir, dataset_sequences):
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
            if _end_fn > _start_fn:
                dataset_sequences.append((1, video_fn, _start_fn, _end_fn))


def load_kinetics600(data_dir, csv_filepath, dataset_sequences):
    with open(csv_filepath) as _f:
        csvrdr = csv.reader(_f)
        cap = cv2.VideoCapture()

        for row in csvrdr:
            # smoking,EYJlJZuSypI,1,11,train
            class_name, id, start_sec, end_sec, train_test = row

            video_fn = os.path.join(data_dir, train_test, "%s.mp4" % id)
            # cap = cv2.VideoCapture(data_dir + "/" + video_fn)
            cap.open(data_dir + "/" + video_fn)

            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if h > w:
                print("Vertical video? %s. skipping" % video_fn)
                cap.release()
                continue

            cap.release()

            _start_fn = int(fps * float(start_sec))
            _end_fn = int(fps * float(end_sec))
            # print(id, _start_fn, _end_fn)
            if _end_fn > _start_fn:
                if "smoking" == class_name:
                    dataset_sequences.append((1, video_fn, _start_fn, _end_fn))
                else:
                    dataset_sequences.append((0, video_fn, _start_fn, _end_fn))
        del cap


class I3DFusionSequence(Sequence):

    def __init__(self, data_dir, train_txt: str, batch_size: int = 32, input_hw=(224, 224), show=False,
                 num_frames_in_sequence: int = 16, only_temporal=False, only_spacial=False):
        self.only_spacial = only_spacial
        self.only_temporal = only_temporal

        self.input_hw = input_hw
        self.num_frames = num_frames_in_sequence
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.show = show
        self.image_augmentation = None
        # self.image_augmentation = augmentation.ImageAugmentation()

        dataset_seq = train_txt + "_dataset_seq.json"

        if os.path.isfile(dataset_seq):
            with open(dataset_seq, 'r') as _f:
                self.dataset_seq = json.load(_f)
        else:
            self.dataset_seq = load_dataset_sequences(data_dir, train_txt)
            with open(dataset_seq, 'w') as _f:
                json.dump(self.dataset_seq, _f)

        assert len(self.dataset_seq) > 0, "empty dataset. something is wrong"

        # all_seq = [  (1, video_fn, cls_seq), ..., ...  ]
        # dataset_seq = [  (1, video_fn, start_frame, end_frame), ..., ...  ]
        self.dataset_seq = list(filter(lambda e: e[3] - e[2] >= num_frames_in_sequence, self.dataset_seq))

        shuffle(self.dataset_seq)
        pos = list(filter(lambda _x: _x[0] == 1, self.dataset_seq))
        neg = list(filter(lambda _x: _x[0] == 0, self.dataset_seq))
        print("Samples with %d frames or more: %d" % (num_frames_in_sequence, len(self.dataset_seq)))
        print("Pos Samples: %d; frames: %d" % (len(pos),
                                               functools.reduce(lambda y, x: y + (x[3] - x[2]),
                                                                pos, 0)))
        print("Neg Samples: %d; frames %05d" % (len(neg),
                                                functools.reduce(lambda y, x: y + (x[3] - x[2]),
                                                                 neg, 0)))

    def __len__(self):
        return int(numpy.ceil(len(self.dataset_seq) / self.batch_size))

    def __getitem__(self, index):
        s = index * self.batch_size
        e = min(len(self.dataset_seq), s + self.batch_size)

        xrgb_batch = numpy.zeros(shape=(self.batch_size, self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        xflow_batch = numpy.zeros(shape=(self.batch_size, self.num_frames, self.input_hw[0], self.input_hw[1], 2))
        ybatch = numpy.zeros(shape=(self.batch_size, 2))

        for ii in range(s, e):
            i = ii - s
            xrgb, xflow, y = self.get_one_xy(ii, no_oflow=self.only_spacial)
            xrgb_batch[i, :, :, :, :] = xrgb
            xflow_batch[i, :, :, :, :] = xflow
            ybatch[i, :] = y

        if self.only_spacial:
            return xrgb_batch, ybatch
        elif self.only_temporal:
            return xflow_batch, ybatch
        else:
            return [xrgb_batch, xflow_batch], ybatch

    def get_one_xy(self, index, no_oflow=False):
        if self.image_augmentation:
            self.image_augmentation.renew()  # change random sequence for imgaug

        datum = self.dataset_seq[index]

        cls_id, v_file, start_frame, end_frame = datum

        if self.show:
            print(index, v_file)

        # the sequence may be long so start every time from a random frame
        rand_start_frame = randint(start_frame, end_frame - self.num_frames)

        video_path = os.path.join(self.data_dir, v_file)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, rand_start_frame)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        fn_counter = itertools.count()
        xframes = numpy.zeros(shape=(self.num_frames, self.input_hw[0], self.input_hw[1], 3))
        xflow = numpy.zeros(shape=(self.num_frames, self.input_hw[0], self.input_hw[1], 2))

        prev_gray = None

        x = randint(0, 64)  # jitter crop for the sequence
        y = randint(0, 64)  # jitter crop for the sequence
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

            if no_oflow:
                cur_flow = numpy.zeros(shape=(self.input_hw[0], self.input_hw[1], 2))
            else:
                cur_flow = calc_flow(gray, prev_gray)
            xflow[fn, :, :, :] = cur_flow
            prev_gray = gray

            if self.show:
                cv2.imshow("spacial clsid%d" % (cls_id), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                if not no_oflow:
                    hsv = numpy.zeros_like(rgb, dtype=numpy.uint8)
                    mag, ang = cv2.cartToPolar(cur_flow[..., 0], cur_flow[..., 1])
                    hsv[..., 0] = ang * 180 / numpy.pi / 2
                    hsv[:, :, 1] = 255
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    cv2.imshow("temporal clsid%d" % (cls_id), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

                cv2.waitKey(25)

            rgb = rgb / 127.5 - 1
            xframes[fn, :, :, :] = rgb

        cap.release()
        del cap
        del prev_gray

        y = numpy.zeros(shape=(1, 2))
        y[0][cls_id] = 1.
        return xframes, xflow, y


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

    seq = I3DFusionSequence("/Volumes/bstorage/datasets/vg_smoke/", "train.txt",
                            input_hw=(224, 224), batch_size=32, num_frames_in_sequence=32,
                            only_temporal=True,
                            show=True)
    print("total batches with samples", len(seq))
    # val = C3DSequence("/Volumes/bstorage/datasets/vg_smoke/", "validate.txt")

    for i in range(len(seq)):
        print(seq[i])
