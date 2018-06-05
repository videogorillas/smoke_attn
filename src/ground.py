import sys

import cv2
import numpy

from utils import yield_frames, truth_from_csv

if __name__ == '__main__':
    in_csv = "/blender/storage/datasets/vg_smoke/valid/basic_inst/basic-instinct_truth.csv"
    by_frame = truth_from_csv(truth_csv=in_csv, num_classes=2)
    for fn, bgr in yield_frames("/blender/storage/datasets/vg_smoke/valid/basic_inst/basic-cropped.mp4"):
        bgr = cv2.resize(bgr, dsize=(640, 480))
        cls_id = numpy.argmax(by_frame[fn])
        cv2.imshow("frame %d" % cls_id, bgr)
        code = cv2.waitKey(25)
        if code == 27:
            sys.exit(1)
