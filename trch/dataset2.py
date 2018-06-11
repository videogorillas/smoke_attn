import os
from random import shuffle, randint

import cv2
import numpy as np
import torch.utils.data as data

from utils import yield_frames


def rgb_and_flows(input_shape_hw: tuple, video_file: str, flows_count: int = 10):
    drop_first_n_frames = randint(0, 25 * 2)
    old_gray = None
    rgb = None
    crop_x1 = randint(0, 32)
    crop_x2 = randint(0, 32)
    crop_y1 = randint(0, 32)
    crop_y2 = randint(0, 32)

    skip_n_frames = 2
    flows_mag_ang = np.zeros(shape=(299, 299, flows_count * 2))
    for fn, bgr in yield_frames(video_file):
        h, w, c = bgr.shape
        bgr = bgr[crop_y1:h - crop_y2, crop_x1:w - crop_x2]

        bgr = cv2.resize(bgr, dsize=input_shape_hw)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if fn > drop_first_n_frames:
            if fn % skip_n_frames != 0:
                continue

            if fn < skip_n_frames * flows_count + 1:
                flow_frame = 2 * int(fn / skip_n_frames) - 2

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


class VideoSeqenceDataset(data.Dataset):
    def __init__(self, data_dir: str, neg_txt: str, pos_txt: str, input_shape_hwc: tuple,
                 show_data=False, only_temporal=False, only_spacial=False):
        self.only_temporal = only_temporal
        self.only_spacial = only_spacial

        self.show = show_data
        self.input_shape_hwc = input_shape_hwc
        self.data_dir = data_dir

        with open(os.path.join(data_dir, neg_txt), 'r') as list_f:
            items = list(map(lambda l: [l.strip(), 0], list_f.readlines()))

        with open(os.path.join(data_dir, pos_txt), 'r') as list_f:
            p = list(map(lambda l: [l.strip(), 1], list_f.readlines()))
            items.extend(p)

        self.file_and_clsid = items
        shuffle(self.file_and_clsid)

    def __getitem__(self, index):
        flows_count: int = 10

        y = np.zeros(shape=(2))
        gif, cls_id = self.file_and_clsid[index]
        f = os.path.join(self.data_dir, gif)
        x_rgb, x_flows = rgb_and_flows(input_shape_hw=self.input_shape_hwc[:2], video_file=f, flows_count=flows_count)

        y[cls_id] = 1.

        return (np.array(x_rgb), np.array(x_flows)), y

    def __len__(self):
        return len(self.file_and_clsid)
