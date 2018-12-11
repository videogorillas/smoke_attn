import itertools

import cv2
import numpy

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
    cap = cv2.VideoCapture("/Volumes/SD128/testvideo/Lisa_Smoke_scene_Jolie.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/basic_inst/basic.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/dron_video/DayFlight.mpg")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/T0076371.mp4.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/Bikepark-LK0GSwofipY.mp4")

    # cap = cv2.VideoCapture("/Volumes/SD128/macg/BX138_SRNA_03.mov")
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 400)

    # cap = cv2.VideoCapture("/Volumes/SD128/macg/MACG_S02_Ep024_ING_5764188.mov")
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
    temporal_size = 4 * 10
    seq_twhc = numpy.zeros(shape=(temporal_size, w, h, c), dtype=numpy.uint8)

    t_delta = 4
    fn_iter = itertools.count(0)
    t_iter = itertools.count(0)
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
        _bgr = bgr_whc.copy()
        seq_twhc[t, :, :, :] = bgr_whc
        temporal_edges = numpy.zeros_like(bgr_whc)
        temporal_corners = numpy.zeros_like(bgr_whc)
        # cv2.imshow("frame", _bgr)
        cv2.imshow("frame", seq_twhc[max(0, t - t_delta), :, :, :])

        for _y in range(0, h, 2):
            slice_X = seq_twhc[:, :, _y, :]
            gray_slice_X = cv2.cvtColor(slice_X, cv2.COLOR_BGR2GRAY)

            mean = 42
            edges_X = cv2.Canny(gray_slice_X, mean * 0.66, mean * 1.33, apertureSize=3)
            cv2.imshow("edge X", edges_X)
            temporal_edges[:, _y, 2] = edges_X[max(0, t - t_delta), :]

            dst = cv2.cornerHarris(edges_X, blockSize=2, ksize=3, k=0.04)
            dst[dst > 0.42 * 1e-1] = 255
            temporal_corners[:, _y, 2] = dst[max(0, t - t_delta), :]
            cv2.imshow("space-timeX", slice_X)
            cv2.imshow("dstX", dst)

        for _x in range(0, w, 2):
            slice_Y = seq_twhc[:, _x, :, :]

            gray_slice_Y = cv2.cvtColor(slice_Y, cv2.COLOR_BGR2GRAY)

            # Canny
            mean = 42
            edges_Y = cv2.Canny(gray_slice_Y, mean * 0.66, mean * 1.33, apertureSize=3)
            cv2.imshow("edge Y", edges_Y)

            temporal_edges[_x, :, 1] = edges_Y[max(0, t - t_delta), :]

            dst = cv2.cornerHarris(edges_Y, blockSize=2, ksize=3, k=0.04)
            dst[dst > 0.42 * 1e-1] = 255
            # dst[dst >1] = 255
            temporal_corners[_x, :, 1] = dst[max(0, t - t_delta), :]
            cv2.imshow("dstY", dst)

            cv2.imshow("space-timeY", slice_Y)

        cv2.imshow("temporal edges", temporal_edges)
        cv2.imshow("temporal corners", temporal_corners)
        # cv2.waitKey(0)

        # Harris
        # harrisCorner_Y = cv2.cornerHarris(gray_slice_Y, blockSize=2, ksize=3, k=0.0042)
        # harrisCorner_Y[harrisCorner_Y > 0.42 * 1e-6] = 255
        # # harrisCorner_Y[harrisCorner_Y <= 0.42 * 1e-6] = 0
        # cv2.imshow("keypoints", harrisCorner_Y)

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

        cv2.waitKey(1)

        if fn == 0:
            cv2.waitKey(0)

        if t == temporal_size - 1:
            pass
            # cv2.waitKey(0)
            # 
            # for _x in range(0, w, 16):
            #     cv2.line(_bgr, (0, _x), (h, _x), (255, 255, _x))
            #     cv2.imshow("frame", _bgr)
            # 
            #     slice_Y = seq_twhc[:, _x, :, :]
            #     gray_slice_Y = cv2.cvtColor(slice_Y, cv2.COLOR_BGR2GRAY)
            #     mean = 42
            # 
            #     edges_Y = cv2.Canny(gray_slice_Y, mean * 0.66, mean * 1.33)
            # 
            #     cv2.imshow("edge Y", edges_Y)
            #     cv2.imshow("space-timeY", slice_Y)
            #     cv2.waitKey(25)
            # 
            # for _y in range(0, h, 16):
            #     cv2.line(_bgr, (_y, 0), (_y, w), (255, 255, _y))
            #     cv2.imshow("frame", _bgr)
            # 
            #     slice_X = seq_twhc[:, :, _y, :]
            #     gray_slice_X = cv2.cvtColor(slice_X, cv2.COLOR_BGR2GRAY)
            #     mean = 42
            #     edges_X = cv2.Canny(gray_slice_X, mean * 0.66, mean * 1.33)
            # 
            #     cv2.imshow("edge X", edges_X)
            #     cv2.imshow("space-timeX", slice_X)
            #     cv2.waitKey(25)

            # seq_twhc[:, :, :, :] = 0
            seq_twhc[:-1] = seq_twhc[1:]
            seq_twhc[-1] = 0
            t_iter = itertools.count(t)
