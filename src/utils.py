import itertools

import cv2


def yield_frames(input_video_url: str, start_msec: float = 0):
    cap = cv2.VideoCapture(input_video_url)

    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    try:
        fn_iter = itertools.count(start=0)
        while cap.isOpened():
            fn = next(fn_iter)
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret, frame = cap.read()
            if not ret:
                break
            yield (fn, frame)

    finally:
        cap.release()
