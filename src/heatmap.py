import sys

import cv2
import matplotlib.pyplot as plt
import numpy
from keras.models import load_model

from dataset import yield_frames

feature_params = dict(maxCorners=0,
                      qualityLevel=0.2,
                      minDistance=3,
                      blockSize=2)
lk_params = dict(winSize=(5, 5),
                 maxLevel=8,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__ == '__main__':
    plt.ioff()

    m = load_model('/Volumes/bstorage/vgmodels/smoking/inception_v3.h5')

    results = []
    xbatch = numpy.zeros((1, 299, 299, 3))
    # "/Volumes/SD128/getty_1525701638930.mp4"):
    # "/Volumes/bstorage/datasets/kinetics-600/train/3AmZxSwEPoQ.mp4"):
    # "/Volumes/bstorage/datasets/smoking_videos/A_Beautiful_Mind_1_smoke_h_nm_np1_fr_goo_8.avi"):
    # "/Volumes/SD128/testvideo/basic_inst/basic.mp4"):
    url = "/Volumes/bstorage/datasets/smoking_scenes/020 - Keira Knightley and Anna Friel Smoking in London Boulevard.mp4"
    for fn, bgr in yield_frames(url):
        print("fn:", fn)

        bgr = cv2.resize(bgr, (420, 420))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(gray, **feature_params)
        if p0 is None:
            p0 = []

        print(len(p0))

        bgr_sq = cv2.resize(bgr, (299, 299))
        rgb = cv2.cvtColor(bgr_sq, cv2.COLOR_BGR2RGB)
        x = rgb / 127.5 - 1
        xbatch[0] = x

        ybatch = m.predict(xbatch)
        for y in ybatch:
            print("y:", y.round(2))
            results.append(y.round(2))

            gray = numpy.zeros((42, fn + 1))

            for f in p0:
                cv2.circle(bgr, tuple(f[0]), 5, (255, 255, 255))

            # bin = numpy.array(y > 0.8).astype(int)
            # i = numpy.argmax(y)
            i = numpy.argmax(y)
            if y[i] < 0.9:
                plt.subplot(331)
                plt.title('drop')
                plt.imshow(rgb)

                # cv2.imshow("drop", bgr)
                print(fn, "DROP", len(p0))
            elif i == 1:
                plt.subplot(332)
                plt.title('smoke')
                plt.imshow(rgb)

                # cv2.imshow("smoking", bgr)
                print(fn, "SMOKE", len(p0))
                gray[:, :] = numpy.array(list(map(lambda r: r[1], results))) * 256
            else:
                plt.subplot(333)
                plt.title('no smoke')
                plt.imshow(rgb)

                # cv2.imshow("no smoking", bgr)
                print(fn, "NO SMOKE", len(p0))
                # gray[:, :] = numpy.array(list(map(lambda r: r[0], results))) * 256

            # plt.subplot(334)
            # plt.title("features")

            plt.subplot(334)
            plt.plot(list(range(0, fn + 1)), list(map(lambda r: r[1], results)), numpy.repeat([0.9], fn + 1), numpy.repeat([0.1], fn + 1))
            plt.yscale('linear')
            plt.title('smoking')
            plt.grid(True)

            plt.show()

            # print(gray)
            # print(gray.shape)

            # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            # cv2.imshow('gray', gray)
            # cv2.imshow('hist', hist)

        code = cv2.waitKey(25)
        if code == 27:
            sys.exit(1)

        xbatch = numpy.zeros((1, 299, 299, 3))
