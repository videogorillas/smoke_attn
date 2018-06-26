import itertools
import sys

import cv2
import numpy as np


def yield_3squares_from_frame_cv(input_video_url: str):
    try:
        cap = cv2.VideoCapture(input_video_url)

        fn_iter = itertools.count(start=0)
        while cap.isOpened():
            fn = next(fn_iter)
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret, frame = cap.read()
            if not ret:
                break
            yield (fn, frame)

            # 
            # h, w, c = frame.shape
            # x = h / input_height
            # new_w = int(w / x)
            # 
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img_hwc = cv2.resize(rgb, (new_w, input_height))
            # 
            # img1 = img_hwc[:, 0:input_height, :]
            # img2 = img_hwc[:, (new_w - input_height):new_w:, :]
            # 
            # center_w = int(new_w / 2)
            # img3 = img_hwc[:, center_w - int(input_height / 2) - 1: center_w + int(input_height / 2), :]
            # 
            # yield [fn, img_hwc, img1, img2, img3]

    finally:
        cap.release()


# mouse callback function
drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
ex, ey = -1, -1
img = None


def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    global ex, ey

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                # cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
                pass
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
            ex = x
            ey = y
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('cur', img)


if __name__ == '__main__':
    # input_video_url = "/Users/chexov/testvideo/smoke_scene_in_the_movies.mp4"
    input_video_url = "/Users/chexov/testvideo/Lisa_Smoke_scene_Jolie.mp4"

    cv2.namedWindow('cur')
    cv2.setMouseCallback('cur', draw_circle)

    homography = [
        [7.6285898e-01, -2.9922929e-01, 2.2567123e+02],
        [3.3443473e-01, 1.0143901e+00, -7.6999973e+01],
        [3.4663091e-04, -1.4364524e-05, 1.0000000e+00]
    ]

    homography = np.array(homography)
    akaze = cv2.AKAZE_create()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # matcher(NORM_HAMMING);
    nn_matches = []

    kpts0 = None
    kpts1 = None
    kpts2 = None

    desc0 = None
    desc1 = None
    desc2 = None

    mask = None

    cut_gray = None
    prev_gray = None
    im3 = None

    good_features_buffer = []

    for fn, bgr in yield_3squares_from_frame_cv(input_video_url):
        print("fn", fn)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if fn == 0:
            prev_gray = gray

            cut_gray = gray
            mask = np.zeros_like(gray)
            kpts1, desc1 = akaze.detectAndCompute(gray, None)
            kpts0, desc0 = akaze.detectAndCompute(cut_gray, None)
            continue

        kpts2, desc2 = akaze.detectAndCompute(gray, None)

        # print(desc1)
        # print(desc2)

        # nn_matches = matcher.knnMatch(desc1, desc2, k=2)
        nn_matches = matcher.knnMatch(desc0, desc2, k=2)

        good = []
        for m, n in nn_matches:
            ratio = 0.7
            if m.distance < ratio * n.distance:
                good.append([m])

        print("good=", len(good))
        good_features_buffer.append(good)

        if (len(good_features_buffer) > 5):
            _prev2, _prev1, _cur, _next1, _next2 = good_features_buffer[-5:]
            prev2, prev1, cur, next1, next2 = len(_prev2), len(_prev1), len(_cur), len(_next1), len(_next2)
            sd = cur * 2 - (prev1 + prev2) / 2 - (next1 + next2) / 2
            if prev1 > 0:
                sd = sd / prev1
            print("sd=", sd, )

            if abs(sd) > 9.:
                print("CUT SD")
                cv2.waitKey(0)

        # if len(good) < 42:
        print(fn % 25)
        if len(good) < 42:
            print("CUT")
            # cv2.waitKey(0)

        if fn % 25 == 0 or len(good) < 42:
            cut_gray = gray
            kpts0, desc0 = akaze.detectAndCompute(gray, None)
            continue

        im3 = cv2.drawMatchesKnn(cut_gray, kpts0, gray, kpts2, good[:42], None,
                                 matchColor=None, matchesMask=None,
                                 flags=2)
        prev_gray = gray
        cv2.imshow("frame", im3)
        if cv2.waitKey(25) & 0xFF == 27:
            sys.exit(1)
