import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import average_precision_score, precision_recall_curve

from utils import yield_frames, truth_from_csv


def main():
    truth_f = "/Volumes/bstorage/home/chexov/smoking/basic_inst/basic-instinct_truth.csv"

    truth_by_frame = truth_from_csv(truth_f)

    jl = "/Volumes/bstorage/home/chexov/cls/classifiers/basic-cropped.mp4.jsonl/result.jsonl"
    jl = "/Volumes/bstorage/home/chexov/cls/classifiers/500_500_basic-cropped.mp4.jsonl/result.jsonl"

    with open(jl, "r") as _f:
        jsons = map(lambda s: s.strip(), _f.readlines())
        results_by_frame = list(map(lambda j: json.loads(j), jsons))

    url = "/blender/storage/home/chexov/smoking/basic_inst/basic-cropped.mp4"

    predictions = []
    truth = []

    for fn, bgr in yield_frames(url):
        _fn, y = results_by_frame[fn]
        true_y = truth_by_frame[fn]
        assert _fn == fn
        predictions.append(y)
        truth.append(true_y)

        # print(y, true_y)
        i = numpy.argmax(numpy.array(y))
        true_i = numpy.argmax(numpy.array(true_y))

        show = True
        if show:
            if y[i] > 0.9:
                cv2.imshow('i=%d;true=%d' % (i, true_i), bgr)
                code = cv2.waitKey(50)
                if code == 27:
                    sys.exit(1)

    predictions = numpy.array(predictions)
    truth = numpy.array(truth)

    # predictions = numpy.array(predictions > 0.9).astype(int)

    # print(truth.shape)
    # print(predictions.shape)
    # precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true=truth, y_pred=predictions,
    #                                                                           average='micro')
    # print(precision, recall)

    p1 = list(map(lambda p: p[0], predictions))
    t1 = list(map(lambda p: p[0], truth))

    p1 = numpy.array(p1)
    t1 = numpy.array(t1)

    average_precision = average_precision_score(t1, p1)
    precision, recall, thresholds = precision_recall_curve(t1, p1)
    print("thresholds=", thresholds)

    plt.subplot(211)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    p1 = list(map(lambda p: p[1], predictions))
    t1 = list(map(lambda p: p[1], truth))

    p1 = numpy.array(p1)
    t1 = numpy.array(t1)

    average_precision = average_precision_score(t1, p1)
    precision, recall, _ = precision_recall_curve(t1, p1)

    plt.subplot(212)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


if __name__ == '__main__':
    main()
