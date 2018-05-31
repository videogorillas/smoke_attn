#!/usr/bin/env python
import json
import os
import sys
from random import shuffle, randint

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
                 show_data=False, only_temporal=False, only_spacial=False):
        self.only_temporal = only_temporal
        self.only_spacial = only_spacial
        self.batch_size = batch_size

        self.show = show_data
        self.input_hwc = input_shape_hwc
        self.data_dir = data_dir

        with open(os.path.join(data_dir, neg_txt), 'r') as list_f:
            items = list(map(lambda l: [l.strip(), 0], list_f.readlines()))

        with open(os.path.join(data_dir, pos_txt), 'r') as list_f:
            p = list(map(lambda l: [l.strip(), 1], list_f.readlines()))
            items.extend(p)

        self.file_and_clsid = items
        shuffle(self.file_and_clsid)

        self.len = int(len(self.file_and_clsid) / batch_size)

    def __getitem__(self, index):
        xx, y_batch = self.rgb_and_flows_batch(index)
        rgb = xx[0]
        flow = xx[1]

        if self.only_temporal:
            return flow, y_batch
        elif self.only_spacial:
            return rgb, y_batch
        else:
            return xx, y_batch

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

    def rgb_and_flows(self, video_file: str, flows_count: int = 10,
                      drop_every_n_frame: int = 2,
                      drop_first_n_frames: int = -1):
        old_gray = None
        rgb = None
        crop_x1 = randint(0, 32)
        crop_x2 = randint(0, 32)
        crop_y1 = randint(0, 32)
        crop_y2 = randint(0, 32)

        if drop_first_n_frames < 0:
            drop_first_n_frames = randint(0, 25 * 2)

        flows_mag_ang = np.zeros(shape=(self.input_hwc[1], self.input_hwc[0], flows_count * 2))
        for fn, bgr in yield_frames(video_file):
            h, w, c = bgr.shape
            bgr = bgr[crop_y1:h - crop_y2, crop_x1:w - crop_x2]

            bgr = cv2.resize(bgr, dsize=(self.input_hwc[0], self.input_hwc[1]))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if fn > drop_first_n_frames:
                if fn % drop_every_n_frame != 0:
                    continue

                if fn < drop_every_n_frame * flows_count + 1:
                    flow_frame = 2 * int(fn / drop_every_n_frame) - 2

                    flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flows_mag_ang[:, :, flow_frame] = mag
                    flows_mag_ang[:, :, flow_frame + 1] = ang
                    old_gray = gray
                else:
                    break
            else:
                old_gray = gray

        return rgb, flows_mag_ang

    def best_frame_for_cls_id(self, drop_every_n_frame, flows_count, json_lf):
        with open(json_lf) as f:
            predictions_by_frame = [json.loads(s.strip()) for s in f]
        total_frames = len(predictions_by_frame)
        class1_prob = np.array([x[1][1] for x in predictions_by_frame])
        rand = np.random.uniform(0, 1, total_frames)
        # find center frame, probably max of the sequence
        center_frame_num = int(np.argmax(class1_prob * rand))
        # check that first frame is > 0
        drop_first_n_frames = max(1, center_frame_num - flows_count * drop_every_n_frame // 2)
        # check that last frame is <= n
        drop_first_n_frames += min(0, total_frames - center_frame_num - flows_count * drop_every_n_frame // 2)
        return drop_first_n_frames

    def rgb_and_flows_batch(self, index):
        s = index * self.batch_size

        flows_count: int = 10
        xrgb_batch = []
        xflow_batch = []
        y_batch = np.zeros(shape=(self.batch_size, 2))
        for i in range(0, self.batch_size):
            gif, cls_id = self.file_and_clsid[s + i]
            video_path = os.path.join(self.data_dir, gif)

            drop_every_n_frame = 2
            drop_first_n_frames = -1

            # if cls_id == 1:
            # hack-hack: choose best frame for smoking based on photo trained cnnmodel
            json_lf = os.path.join(self.data_dir, "jsonl", gif.replace("/mp4/", "/"), 'result.jsonl')
            if os.path.isfile(json_lf):
                drop_first_n_frames = self.best_frame_for_cls_id(drop_every_n_frame, flows_count, json_lf)
                # print("JSONL best frame is %d : %s" % (drop_first_n_frames, gif))

            x_rgb, x_flows = self.rgb_and_flows(video_path,
                                                drop_every_n_frame=drop_every_n_frame,
                                                drop_first_n_frames=drop_first_n_frames,
                                                flows_count=flows_count)

            xrgb_batch.append(x_rgb)
            xflow_batch.append(x_flows)
            y_batch[i][cls_id] = 1.

        return [np.array(xrgb_batch), np.array(xflow_batch)], y_batch


def test():
    data_dir = "/bstorage/datasets/smoking/gifs/"
    data_dir = "/blender/storage/datasets/vg_smoke"
    seq = SmokeGifSequence(data_dir, neg_txt='negatives.txt', pos_txt='positives.txt', input_shape_hwc=(300, 299, 3),
                           only_spacial=False, only_temporal=False)
    # bseq = BatchSeq(seq, batch_size=16)

    for i in range(len(seq)):
        # x_rgb_b, x_flows_b, y_b = seq[i]
        xx, y_b = seq[i]
        x_rgb_b, x_flows_b = xx

        for j in range(len(x_rgb_b)):
            x_rgb, y = x_rgb_b[j], y_b[j]
            x_flows = x_flows_b[j]

            bgr = cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR)

            hsv = np.zeros_like(x_rgb)
            mag = x_flows[:, :, 0]
            ang = x_flows[:, :, 1]
            hsv = seq.flow_to_hsv(dst=hsv, mag_ang=(mag, ang))
            flow_mask = 255 - cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # cv2.imshow("bgr", bgr)
            cls_id = np.argmax(y)
            cv2.imshow("rgb+flow %d" % cls_id, bgr - flow_mask)

            c = cv2.waitKey(0)
            if c == 27:
                sys.exit(1)


if __name__ == '__main__':
    test()
