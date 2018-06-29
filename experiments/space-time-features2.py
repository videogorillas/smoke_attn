import itertools

import cv2
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    akaze = cv2.AKAZE_create()
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
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 54608)  # people walking scene
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 56426)  # propeller
    cap.set(cv2.CAP_PROP_POS_FRAMES, 62788)  # dark cut
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 66288)  # person walking  across the scene
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 5000) 
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 6278)  

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = 320
    h = 480
    c = 3

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    # tempolength = int(fps) * 3
    temporal_size = 24 * 11
    seq_twhc = numpy.zeros(shape=(temporal_size, w, h, c), dtype=numpy.uint8)

    fn_iter = itertools.count(0)
    t_iter = itertools.count(0)
    pyplot.title('H')
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # cv2.namedWindow("harris")
    # harrisThreshold = 42
    # cv2.createTrackbar("Threshold", "harris", harrisThreshold, 255,)
    while cap.isOpened():
        ret, bgr_whc = cap.read()
        if not ret:
            break
        fn = next(fn_iter)
        t = next(t_iter)
        print(fn, t)

        bgr_whc = cv2.resize(bgr_whc, (h, w))
        seq_twhc[t, :, :, :] = bgr_whc

        _bgr = bgr_whc.copy()
        _x = int(w / 2)
        _y = int(h / 2)
        cv2.line(_bgr, (0, _x), (h, _x), (255, 255, 255))
        cv2.line(_bgr, (_y, 0), (_y, w), (255, 255, 255))
        cv2.imshow("frame", _bgr)

        # Autoexposure
        # _norm = _bgr.copy()
        # print(_bgr.mean())
        # if _bgr.mean() < 16:
        #     _norm = cv2.normalize(_bgr, _norm, alpha=0, beta=255 * int(_bgr.mean()), norm_type=cv2.NORM_MINMAX)
        # 
        # sift_kp = sift.detect(cv2.cvtColor(_norm, cv2.COLOR_BGR2GRAY), None)
        # _norm = cv2.drawKeypoints(_norm, sift_kp, None)
        # sift_kp = sift.detect(cv2.cvtColor(_bgr, cv2.COLOR_BGR2GRAY), None)
        # _bgr = cv2.drawKeypoints(_bgr, sift_kp, None)
        # cv2.imshow("auto exposure frame", numpy.vstack([_norm, _bgr]).astype(numpy.uint8))

        slice_Y = seq_twhc[:, _x, :, :]
        slice_X = seq_twhc[:, :, _y, :]

        gray_slice_X = cv2.cvtColor(slice_X, cv2.COLOR_BGR2GRAY)
        gray_slice_Y = cv2.cvtColor(slice_Y, cv2.COLOR_BGR2GRAY)

        # Canny
        mean = 42
        edges_X = cv2.Canny(gray_slice_X, mean * 0.66, mean * 1.33, apertureSize=3)
        edges_Y = cv2.Canny(gray_slice_Y, mean * 0.66, mean * 1.33, apertureSize=3)
        cv2.imshow("edge Y", edges_Y)
        cv2.imshow("edge X", edges_X)

        # Harris
        harrisCorner_Y = cv2.cornerHarris(gray_slice_Y, blockSize=2, ksize=3, k=0.0042)
        harrisCorner_Y[harrisCorner_Y > 0.42 * 1e-6] = 255
        # harrisCorner_Y[harrisCorner_Y <= 0.42 * 1e-6] = 0
        cv2.imshow("keypoints", harrisCorner_Y)

        # dst_norm = numpy.zeros_like(harrisCorner_Y)
        # cv2.normalize(harrisCorner_Y, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
        # cv2.imshow("harris", dst_norm_scaled)

        # HoughLines
        # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (142, 1))
        # cv2.imshow("horizontalStructure", horizontalStructure)
        # cv2.waitKey(1)
        # lines = cv2.HoughLinesP(edges_Y, 1, numpy.pi / 180, 100, minLineLength=100, maxLineGap=10)
        # print(lines)
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         slice_Y = cv2.line(slice_Y, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # SIFT
        # sift_kp = sift.detect(gray_slice_Y, None)
        # img = numpy.zeros_like(gray_slice_Y)
        # slice_Y = cv2.drawKeypoints(slice_Y, sift_kp, None)

        #
        cv2.imshow("space-timeY", slice_Y)
        cv2.imshow("space-timeX", slice_X)
        cv2.waitKey(1)

        if fn == 0:
            cv2.waitKey(0)

        if t == temporal_size - 1:
            cv2.waitKey(0)

            for _x in range(0, w, 16):
                cv2.line(_bgr, (0, _x), (h, _x), (255, 255, _x))
                cv2.imshow("frame", _bgr)

                slice_Y = seq_twhc[:, _x, :, :]
                gray_slice_Y = cv2.cvtColor(slice_Y, cv2.COLOR_BGR2GRAY)
                mean = 42

                edges_Y = cv2.Canny(gray_slice_Y, mean * 0.66, mean * 1.33)

                cv2.imshow("edge Y", edges_Y)
                cv2.imshow("space-timeY", slice_Y)
                cv2.waitKey(25)

            for _y in range(0, h, 16):
                cv2.line(_bgr, (_y, 0), (_y, w), (255, 255, _y))
                cv2.imshow("frame", _bgr)

                slice_X = seq_twhc[:, :, _y, :]
                gray_slice_X = cv2.cvtColor(slice_X, cv2.COLOR_BGR2GRAY)
                mean = 42
                edges_X = cv2.Canny(gray_slice_X, mean * 0.66, mean * 1.33)

                cv2.imshow("edge X", edges_X)
                cv2.imshow("space-timeX", slice_X)
                cv2.waitKey(25)

            seq_twhc[:, :, :, :] = 0
            t_iter = itertools.count(0)
