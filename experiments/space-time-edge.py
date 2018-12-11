import itertools

import cv2
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    akaze = cv2.AKAZE_create()
    TVL1: cv2.DualTVL1OpticalFlow = cv2.DualTVL1OpticalFlow_create()

    sift = cv2.xfeatures2d.SIFT_create()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # cap = cv2.VideoCapture("/Volumes/SD128/getty_1525701638930.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/smoke_scene_in_the_movies.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/shuttle-flip.mp4")
    # cap = cv2.VideoCapture("/Volumes/storage/home/zhukov/bf/storage/BX138_SRNA_03.mov/thumbs/thumbs42.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/LittleMermaid_02_Good.mov")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/gray10sec.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/Lisa_Smoke_scene_Jolie.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/basic_inst/basic.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/dron_video/DayFlight.mpg")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/T0076371.mp4.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/Bikepark-LK0GSwofipY.mp4")

    # cap = cv2.VideoCapture("/Volumes/SD128/macg/BX138_SRNA_03.mov")
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 400)

    cap = cv2.VideoCapture("/Volumes/SD128/macg/MACG_S02_Ep024_ING_5764188.mov")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 54608)  # corridor scene
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 56426)  # propeller
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 62788)  # dark cut
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 66288)  # person walking  across the scene
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 5000) 
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 6278)  
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 58555)  # dialog  
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 23300)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = 320
    h = 480
    c = 3

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    # temporal_size = int(fps) * 3
    # temporal_size = 60
    seq_twhc = numpy.zeros(shape=(temporal_size, w, h, c), dtype=numpy.uint8)

    fn_iter = itertools.count(0)
    t_iter = itertools.count(0)
    pyplot.title('H')
    fig = pyplot.figure()
    ax = Axes3D(fig)

    # cv2.namedWindow("harris")
    # harrisThreshold = 42
    # cv2.createTrackbar("Threshold", "harris", harrisThreshold, 255,)
    prev = None
    while cap.isOpened():
        ret, bgr_whc = cap.read()
        if not ret:
            break

        fn = next(fn_iter)
        t = next(t_iter)
        print(fn, t)

        bgr_whc = cv2.resize(bgr_whc, (h, w))
        if prev is None:
            prev = bgr_whc

        seq_twhc[t, :, :, :] = bgr_whc

        _bgr = bgr_whc.copy()
        temoral_edge = numpy.zeros_like(bgr_whc)

        # cv2.imshow("frame", _bgr)
        for _x in range(0, w, 16):
            slice_Y = seq_twhc[:, _x, :, :]
            gray_slice_Y = cv2.cvtColor(slice_Y, cv2.COLOR_BGR2GRAY)
            mean = 42

            edges_Y = cv2.Canny(gray_slice_Y, mean * 0.66, mean * 1.33)

            temoral_edge[_x, :, 1] = edges_Y[max(0, t - 2), :]
            # _bgr[_x, :, 1] = edges_Y[max(0, t - 3), :]

            cv2.imshow("edge Y", edges_Y)
            cv2.imshow("space-timeY", slice_Y)

        for _y in range(0, h, 16):
            slice_X = seq_twhc[:, :, _y, :]
            gray_slice_X = cv2.cvtColor(slice_X, cv2.COLOR_BGR2GRAY)
            mean = 42
            edges_X = cv2.Canny(gray_slice_X, mean * 0.66, mean * 1.33)
            temoral_edge[:, _y, 1] = edges_X[max(0, t - 2), :]
            # _bgr[:, _y, 1] = edges_X[max(0, t - 3), :]

            cv2.imshow("edge X", edges_X)
            cv2.imshow("space-timeX", slice_X)

        _bgr[temoral_edge > 0] = 255
        # cv2.waitKey(25)
        # cv2.waitKey(0)
        # seq_twhc[:, :, :, :] = 0

        # Optical Flow
        # flow = TVL1.calc(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
        #                     cv2.cvtColor(bgr_whc, cv2.COLOR_BGR2GRAY), None)
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # prev = bgr_whc
        # hsv = numpy.zeros_like(bgr_whc)
        # hsv[..., 0] = ang * 180 / numpy.pi / 2
        # # hsv[..., 0] = 0
        # hsv[..., 1] = 255
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("tvl1", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

        # Harris
        harrisCorner_Y = cv2.cornerHarris(temoral_edge[:, :, 1], blockSize=2, ksize=3, k=0.2)
        # harrisCorner_Y[harrisCorner_Y > 1.] = 255
        harrisCorner_Y[harrisCorner_Y > 0.42 * 1e-6] = 255
        # harrisCorner_Y[harrisCorner_Y <= 0.42 * 1e-6] = 0

        # temoral_edge[harrisCorner_Y > 0] = 255
        # cv2.imshow("keypoints", harrisCorner_Y)

        cv2.imshow("frame", _bgr)
        cv2.imshow("corner", harrisCorner_Y)

        seq_twhc[:-1] = seq_twhc[1:]
        seq_twhc[-1] = 0

        cv2.waitKey(25)
        if t == temporal_size - 1:
            # t_iter = itertools.count(0)
            t_iter = itertools.count(t)
