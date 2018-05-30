#!/usr/bin/env python
import os
import sys
from random import shuffle

import cv2
import numpy as np
from keras.utils import Sequence

from utils import yield_frames


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

        return np.array(x_batch), np.array(y_batch)

    def __len__(self):
        return self.len


class SmokeGifSequence(Sequence):

    def __init__(self, data_dir: str, neg_txt: str, pos_txt: str, input_shape_hwc: tuple, batch_size=16,
                 show_data=False):
        self.batch_size = batch_size

        self.show = show_data
        self.input_shape_hwc = input_shape_hwc
        self.data_dir = data_dir

        with open(neg_txt, 'r') as list_f:
            items = list(map(lambda l: [l.strip(), 0], list_f.readlines()))

        with open(pos_txt, 'r') as list_f:
            p = list(map(lambda l: [l.strip(), 1], list_f.readlines()))
            items.extend(p)

        self.file_and_clsid = items
        shuffle(self.file_and_clsid)

        self.len = int(len(self.file_and_clsid) / batch_size)

    def __getitem__(self, index):
        return self.rgb_and_flows_batch(index)

    def __len__(self):
        return self.len

    @staticmethod
    def flow_to_hsv(dst: np.ndarray, mag_ang: np.ndarray):
        hsv = dst
        mag, ang = mag_ang
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return hsv

    def rgb_and_flows(self, gif_file: str, flows_count: int = 10):
        old_gray = None
        rgb = None

        flows_mag_ang = np.zeros(shape=(flows_count, 2, 299, 299))
        for fn, bgr in yield_frames(gif_file):
            bgr = cv2.resize(bgr, dsize=self.input_shape_hwc[:2])
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if fn > 0:
                if fn < flows_count + 1:
                    flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flows_mag_ang[fn - 1, 0, :, :] = mag
                    flows_mag_ang[fn - 1, 1, :, :] = ang
                else:
                    break
            old_gray = gray

        return rgb, flows_mag_ang

    def rgb_and_flows_batch(self, index):
        s = index * self.batch_size

        flows_count: int = 10
        xrgb_batch = []
        xflow_batch = []
        y_batch = np.zeros(shape=(self.batch_size, 2))
        for i in range(0, self.batch_size):
            gif, cls_id = self.file_and_clsid[s + i]
            x_rgb, x_flows = self.rgb_and_flows(os.path.join(self.data_dir, gif), flows_count=flows_count)

            xrgb_batch.append(x_rgb)
            xflow_batch.append(x_flows)
            y_batch[i][cls_id] = 1.

        return xrgb_batch, xflow_batch, y_batch

    def just_rgb_batch(self, index):
        s = index * self.batch_size

        xrgb_batch = np.zeros(shape=(self.batch_size, 299, 299, 3))
        y_batch = np.zeros(shape=(self.batch_size, 2))
        for i in range(0, self.batch_size):
            gif, cls_id = self.file_and_clsid[s + i]
            x_rgb, x_flows = self.rgb_and_flows(os.path.join(self.data_dir, gif))
            xrgb_batch[i] = x_rgb
            y_batch[i][cls_id] = 1.

        return xrgb_batch, y_batch


def test():
    data_dir = "/bstorage/datasets/smoking/gifs/"
    seq = SmokeGifSequence(data_dir, neg_txt='train_neg.txt', pos_txt='train_pos.txt', input_shape_hwc=(299, 299, 3))
    # bseq = BatchSeq(seq, batch_size=16)

    for i in range(len(seq)):
        # [x_rgb, x_flows], y = seq[i][0]
        x_rgb_b, x_flows_b, y_b = seq[i]
        xrgb_batch, xflow_batch, y_batch = seq[i]

        for j in range(len(x_rgb_b)):
            x_rgb, y = x_rgb_b[j], y_b[j]
            x_flows = x_flows_b[j]

            bgr = cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR)

            hsv = np.zeros_like(x_rgb)
            hsv = seq.flow_to_hsv(dst=hsv, mag_ang=x_flows[0])
            flow_mask = 255 - cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # cv2.imshow("%d rgb %s" % (j, y), bgr - flow_mask)
            cv2.imshow("rgb+flow %d" % j, bgr - flow_mask)

            c = cv2.waitKey(0)
            if c == 27:
                sys.exit(1)


if __name__ == '__main__':
    test()
