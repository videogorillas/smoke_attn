import itertools

import cv2
import numpy

if __name__ == '__main__':
    # cap = cv2.VideoCapture("/Volumes/SD128/getty_1525701638930.mp4")
    cap = cv2.VideoCapture("/Volumes/SD128/testvideo/smoke_scene_in_the_movies.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/shuttle-flip.mp4")
    # cap = cv2.VideoCapture("/Volumes/storage/home/zhukov/bf/storage/BX138_SRNA_03.mov/thumbs/thumbs42.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/LittleMermaid_02_Good.mov")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/gray10sec.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/Lisa_Smoke_scene_Jolie.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/basic_inst/basic.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/dron_video/DayFlight.mpg")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/T0076371.mp4.mp4")
    # cap = cv2.VideoCapture("/Volumes/SD128/testvideo/Bikepark-LK0GSwofipY.mp4")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    tempolength = int(fps) * 20
    tempo = numpy.zeros(shape=(h, tempolength, w, 3), dtype=numpy.uint8)

    fn_iter = itertools.count(0)
    fn_tempo = itertools.count(0)
    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        fn = next(fn_tempo)

        # cv2.imshow("frame", bgr)
        # cv2.waitKey(5)
        for _h in range(1, h):
            # for _h in range(142, 143):
            line = bgr[_h - 1:_h, :, :]

            print(fn)
            tempo[_h, fn, :, :] = bgr[_h, :, :]

            if fn == tempolength - 1:
                fn_tempo = itertools.count(0)
                _bgr = bgr.copy()
                cv2.line(_bgr, (0, _h), (w, _h), (255, 255, 255))
                cv2.imshow("bgr", _bgr)

                cv2.imshow("space-time", tempo[_h])
                cv2.waitKey(5)

                fn_tempo = itertools.count(0)

    for _h in range(1, h):
        cv2.imshow("space-time", tempo[_h])
        cv2.waitKey(5)
