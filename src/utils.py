import csv
import itertools

import cv2
import numpy


def yield_frames_v2(input_video_url: str, start_msec: float = -1, start_frame=-1):
    cap = cv2.VideoCapture(input_video_url)

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if start_msec > -1:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    if start_frame > -1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        fn_iter = itertools.count(start=0)
        while cap.isOpened():
            fn = next(fn_iter)
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret, frame = cap.read()
            if not ret:
                break
            yield (fn, frame, frames)

    finally:
        cap.release()


def yield_frames(input_video_url: str, start_msec: float = -1, start_frame=-1):
    cap = cv2.VideoCapture(input_video_url)

    if start_msec > -1:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    if start_frame > -1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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


def truth_from_csv(truth_csv: str, num_classes: int = 2) -> list:
    truth_by_frame = []
    with open(truth_csv, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            start_fn, end_fn, class_id = row
            start_fn, end_fn, class_id = int(start_fn), int(end_fn), int(class_id)

            for fn in range(start_fn, end_fn + 1):
                y = numpy.zeros(shape=num_classes, dtype=numpy.float)
                y[class_id] = 1.
                truth_by_frame.append(y)
    return truth_by_frame
