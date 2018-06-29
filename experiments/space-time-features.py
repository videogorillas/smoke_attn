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
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/Lisa_Smoke_scene_Jolie.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/basic_inst/basic.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/dron_video/DayFlight.mpg")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/T0076371.mp4.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/Bikepark-LK0GSwofipY.mp4")

    # cap = cv2.VideoCapture("/Volumes/SD128/macg/BX138_SRNA_03.mov")
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 400)

    cap = cv2.VideoCapture("/Volumes/SD128/macg/MACG_S02_Ep024_ING_5764188.mov")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 54608)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = 320
    h = 480

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    # tempolength = int(fps) * 3
    temporal_size = 24 * 5
    temporal_row = numpy.zeros(shape=(h, temporal_size, w, 3), dtype=numpy.uint8)
    temporal_column = numpy.zeros(shape=(w, temporal_size, h, 3), dtype=numpy.uint8)

    fn_iter = itertools.count(0)
    fn_tempo = itertools.count(0)
    while cap.isOpened():
        ret, bgr_whc = cap.read()
        if not ret:
            break

        # bgr = cv2.resize(bgr, (w, h))
        bgr_whc = cv2.resize(bgr_whc, (h, w))

        fn = next(fn_tempo)
        # cv2.imshow("frame", bgr_whc)
        # cv2.waitKey(1)
        # print(fn)

        dst3d = numpy.zeros(())
        for _y in range(1, h):
            # for _h in range(142, 143):
            # line = bgr_whc[_y - 1:_y, :, :]
            line = bgr_whc[:, _y, :]
            temporal_row[_y, fn, :, :] = line

            if fn == temporal_size - 1:
                print(fn)

                fn_tempo = itertools.count(0)
                _bgr = bgr_whc.copy()
                cv2.line(_bgr, (_y, 0), (_y, w), (255, 255, 255))
                cv2.imshow("bgr", _bgr)

                slice__h_ = temporal_row[_y]

                slice_gray = cv2.cvtColor(slice__h_, cv2.COLOR_BGR2GRAY)

                # Harris
                dst = cv2.cornerHarris(slice_gray, 2, 3, 0.04)
                # dst[dst > 0.01 * dst.max()] = 255
                dst[dst > 0.42 * 1e-7] = 255
                cv2.imshow("harris", dst)

                # SIFT
                sift_kp = sift.detect(slice_gray, None)
                img = numpy.zeros_like(slice_gray)
                img = cv2.drawKeypoints(slice_gray, sift_kp, None)

                cv2.imshow("space-timeY", slice__h_)
                cv2.imshow("sift", img)
                cv2.waitKey(0)

                fn_tempo = itertools.count(0)

    for _y in range(1, h):
        cv2.imshow("space-time", temporal_row[_y])
        cv2.waitKey(5)
