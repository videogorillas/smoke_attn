import cv2
import itertools


def yield_frames(input_video_url: str):
    cap = cv2.VideoCapture(input_video_url)
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
