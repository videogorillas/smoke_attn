import itertools
import json
import os
import sys
from os.path import basename
from shutil import move

import cv2
import numpy


def validate_results(video_f, y_by_fn):
    homography = [
        [7.6285898e-01, -2.9922929e-01, 2.2567123e+02],
        [3.3443473e-01, 1.0143901e+00, -7.6999973e+01],
        [3.4663091e-04, -1.4364524e-05, 1.0000000e+00]
    ]
    homography = numpy.array(homography)
    akaze = cv2.AKAZE_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # matcher(NORM_HAMMING);

    cap = cv2.VideoCapture(video_f)
    frame_counter = itertools.count()
    prev_clsid = -1
    prev_gray = None
    prev_desc = None
    prev_kpts = None
    while cap.isOpened():
        fn = next(frame_counter)

        ret, bgr = cap.read()

        if ret is None or bgr is None:
            break

        w, h, c = bgr.shape
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # crop rectangle
        cv2.rectangle(bgr, (32, 32), (h - 32, w - 32), (0, 0, 0), 1)

        if prev_gray is None:
            prev_gray = gray
            prev_kpts, prev_desc = akaze.detectAndCompute(prev_gray, None)

        kpts, desc = akaze.detectAndCompute(gray, None)
        good = good_matches(desc, prev_desc, matcher)
        im3 = cv2.drawMatchesKnn(gray, kpts, prev_gray, prev_kpts, good[:20], None,
                                 matchColor=None, matchesMask=None,
                                 flags=2)
        # cv2.imshow("matches", im3)
        prev_kpts = kpts
        prev_desc = desc

        y = y_by_fn[fn]
        cls_id = numpy.argmax(y[1])
        cls_acc = y[1][cls_id]

        threshold = 0.9
        title = "%s" % basename(video_f)
        if cls_id == 1 and cls_acc > threshold:
            cv2.putText(bgr, "fn%05d %s" % (fn, "(1) smoke? %03f" % cls_acc),
                        (42, 42), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.imshow(title, bgr)

        elif cls_id == 0 and cls_acc > threshold:
            cv2.putText(bgr, "fn%05d %s" % (fn, "(0) NO smoke? %03f" % cls_acc),
                        (42, 42), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.imshow(title, bgr)
        else:
            cv2.putText(bgr, "fn%05d %s" % (fn, "(%d) drop %03f" % (cls_id, cls_acc)),
                        (42, 42), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            cv2.imshow(title, bgr)

        code = cv2.waitKey(25)
        if code == 27:
            cap.release()
            sys.exit(1)
        elif code == 32:
            cls_id = read_cls_id()

        if len(good) < 20 or fn == 0:
            prev_gray = gray

            cv2.putText(bgr, "CUT. Press 'n' to continue",
                        (42, 142), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(bgr, "CUT. Press 'n' to continue",
                        (42, 162), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            cv2.putText(bgr, "CUT. Press 'n' to continue",
                        (42, 182), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.imshow(title, bgr)

            code = -1
            while code != 110:
                code = cv2.waitKey(0)
            cv2.putText(bgr, "WIN! WIN! WIN!", (22, 240),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
            cv2.imshow(title, bgr)

        if len(good) < 20 or prev_clsid != cls_id or cls_acc < threshold - 0.1:
            cls_id = read_cls_id()
            prev_clsid = cls_id
            y = [fn, [0, 0]]
            y[1][cls_id] = 1.

        print("fn: %d; akaze=%d; %s" % (fn, len(good), y))
        yield y
    cap.release()
    cv2.destroyAllWindows()


def read_cls_id():
    cls_id = -1
    while cls_id not in ['0', '1']:
        cls_id = cv2.waitKey(0)
        cls_id = chr(cls_id)
    cls_id = int(cls_id)
    return cls_id


def good_matches(desc, prev_desc, matcher):
    if desc is None:
        return []
    if prev_desc is None:
        return []

    nn_matches = matcher.knnMatch(prev_desc, desc, k=2)
    if nn_matches is None:
        nn_matches = []

    good = []
    for m_n in nn_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        ratio = 0.7
        if m.distance < ratio * n.distance:
            good.append([m])
    return good


def validate_single(machine_json, human_json, video_f):
    y_by_fn = []
    with open(machine_json, 'r') as _f:
        for j in map(lambda l: json.loads(l.strip()), _f.readlines()):
            fn, y = j
            # print(fn, y)
            cls_id = numpy.argmax(y)
            y_by_fn.append(j)

    with open(human_json, 'w') as _f:
        for true_y in validate_results(video_f, y_by_fn):
            fn, y = true_y
            y_by_fn[fn] = y
            _f.write("%s\n" % json.dumps(true_y))
            _f.flush()


def main():
    vg_smoke_dir = '/Volumes/bstorage/datasets/vg_smoke/'
    with open('%s/validate.txt' % vg_smoke_dir, 'r') as _f:
        for mp4 in _f.readlines():
            mp4 = mp4.strip()
            video_f = os.path.join(vg_smoke_dir, mp4)
            human_json = os.path.join(vg_smoke_dir, "jsonl.byhuman", mp4, "result.jsonl")
            machine_json = os.path.join(vg_smoke_dir, "jsonl", mp4, "result.jsonl")

            d = os.path.dirname(human_json)
            os.makedirs(d, exist_ok=True)
            if os.path.isfile(human_json):
                print("EXIST ", human_json)
                continue

            validate_single(machine_json, human_json + ".tmp", video_f)
            move(human_json + ".tmp", human_json)


if __name__ == '__main__':
    main()

    # json_f = "/Volumes/bstorage/datasets/vg_smoke/jsonl/smoking_scenes/072 - Virginia Madsen smoking style.mp4/result.jsonl"
    # video_f = "/Volumes/bstorage/datasets/vg_smoke/smoking_scenes/072 - Virginia Madsen smoking style.mp4"
    # validate_single(json_f, video_f)
