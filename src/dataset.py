#!/usr/bin/env python
import itertools
import os
import sys
from random import shuffle

import cv2
import numpy as np
from keras.utils import Sequence


def yield_frames(input_video_url: str):
    cap = cv2.VideoCapture(input_video_url)
    try:
        fn_iter = itertools.count(start=0)
        while cap.isOpened():
            fn = next(fn_iter)
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret, frame = cap.read()
            if not ret:
                break
            yield (fn, frame)

    finally:
        cap.release()


class BatchSeq(Sequence):

    def __init__(self, seq: Sequence, batch_size=16):
        self.batch_size = batch_size
        self.seq = seq
        self.len = int(len(seq) / batch_size)

    def __getitem__(self, index):
        s = index * self.batch_size
        e = s + self.batch_size

        x_batch = []
        y_batch = []
        for i in range(s, e):
            x, y = self.seq[i]
            x_batch.append(x)
            y_batch.append(y)

        return x_batch, y_batch

    def __len__(self):
        return self.len


class SmokeGifSequence(Sequence):

    def __init__(self, data_dir: str, neg_txt: str, pos_txt: str, input_shape_hwc, show_results=False):
        self.show = show_results
        self.input_shape_hwc = input_shape_hwc
        self.data_dir = data_dir

        with open(neg_txt, 'r') as list_f:
            items = list(map(lambda l: [l.strip(), 0], list_f.readlines()))

        with open(pos_txt, 'r') as list_f:
            p = list(map(lambda l: [l.strip(), 1], list_f.readlines()))
            items.extend(p)

        self.items = items
        shuffle(self.items)

    def frame_and_flow(self, gif_file: str, frames: int = 15):
        old_gray = None
        gray = None
        rgb = None

        for fn, bgr in yield_frames(gif_file):
            bgr = cv2.resize(bgr, dsize=self.input_shape_hwc[:2])
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if fn < 5:
                old_gray = gray
                continue

            if fn % 15:
                continue

        flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(rgb)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return rgb, hsv

    def __getitem__(self, index):
        gif, cls_id = self.items[index]

        rgb, flow_hsv = self.frame_and_flow(os.path.join(self.data_dir, gif))
        x_rgb = rgb
        x_flow = flow_hsv

        y = np.zeros(2)
        y[cls_id] = 1.

        if self.show:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            attn_mask = 255 - cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('hsv', cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR))
            cv2.imshow("frame", bgr)
            cv2.imshow("attention", bgr - attn_mask)

        return [x_rgb, x_flow], y

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':

    data_dir = "/bstorage/datasets/smoking/gifs/"
    seq = SmokeGifSequence(data_dir, neg_txt='train_neg.txt', pos_txt='train_pos.txt', input_shape_hwc=(299, 299, 3))
    bseq = BatchSeq(seq, batch_size=16)

    for i in range(len(seq)):
        # [x_rgb, x_flow], y = seq[i]
        x_b, y_b = bseq[i]
        [x_rgb, x_flow], y = (x_b[0], y_b[0])

        attn_mask = 255 - cv2.cvtColor(x_flow, cv2.COLOR_HSV2BGR)
        cv2.imshow("rgb", cv2.cvtColor(x_rgb - attn_mask, cv2.COLOR_RGB2BGR))

        c = cv2.waitKey(0)
        if c == 27:
            sys.exit(1)
