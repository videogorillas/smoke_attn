from random import shuffle

import cv2
import h5py

from utils import yield_frames


class SmokeGifSpacial(object):

    def __init__(self, data_dir: str, neg_txt: str, pos_txt: str, input_shape_hwc: tuple,
                 show_data=False):
        self.batch_size = 1

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

        self.len = int(len(self.file_and_clsid) / self.batch_size)

    def yield_frames(self, index, every_n=10):
        gif, cls_id = self.file_and_clsid[index]

        for fn, bgr in yield_frames(gif):
            if fn % every_n:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield fn, rgb

    def __len__(self):
        return self.len


if __name__ == '__main__':
    data_dir = "/bstorage/datasets/smoking/gifs/"

    h5 = h5py.File('spacial_smoking_gifs_n10.h5', mode='w')

    seq = SmokeGifSpacial(data_dir, neg_txt='train_neg.txt', pos_txt='train_pos.txt', input_shape_hwc=(299, 299, 3))
