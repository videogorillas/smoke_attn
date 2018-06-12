'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import argparse
import json
from cv2 import DualTVL1OpticalFlow_create as DualTVL1
from itertools import count

import cv2
import numpy as np
from keras.models import load_model

from dataset import calc_flow
from i3d_inception import Inception_Inflated3d

NUM_FRAMES = 16
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

SAMPLE_DATA_PATH = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy'
}

LABEL_MAP_PATH = 'data/label_map.txt'
TVL1 = DualTVL1()


def m2(video_f, show=False):
    m = load_model('/blender/storage/home/chexov/smoke_attn/i3d_kinetics_finetune_v1.0.hdf')

    inputs = []
    flow = []

    cap = cv2.VideoCapture(video_f)
    prev = None
    fn_counter = count()
    fn_startrange = 0
    while cap.isOpened():
        ret, bgr = cap.read()
        fn = next(fn_counter)

        if bgr is None:
            break

        h, w, c = bgr.shape
        ratio = w / h
        w_resized = int(224 * ratio)
        resized = cv2.resize(bgr, (w_resized, 224))
        # cv2.imshow("resized", resized)
        # cv2.waitKey(0)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        w1 = int((w_resized / 2) - 224 / 2)
        rgb = rgb[0:224, w1:w1 + 224, :]

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if prev is None:
            prev = gray

        curr_flow = calc_flow(gray, prev)
        flow.append(curr_flow)
        prev = gray
        if show:
            cv2.imshow("cropped", cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))

        rgb = rgb / 127.5 - 1
        inputs.append(rgb)

        if len(inputs) == NUM_FRAMES:
            x_rgb = np.array([inputs])
            x_flow = np.array([flow])
            y_batch = m.predict([x_rgb, x_flow])

            y = y_batch[0]
            for i in range(fn_startrange, fn):
                yield [i, y.tolist()]

            if show:
                cls_id = np.argmax(y)
                cv2.imshow("%s" % cls_id, resized)
                cv2.waitKey(25)

            inputs = []
            flow = []
            fn_startrange = fn


def m(video_f):
    kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]
    rgb_model = Inception_Inflated3d(
        include_top=True,
        # weights='rgb_imagenet_and_kinetics',
        weights='rgb_kinetics_only',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
        classes=NUM_CLASSES)

    flow_model = Inception_Inflated3d(
        include_top=True,
        # weights='flow_imagenet_and_kinetics',
        weights='flow_kinetics_only',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
        classes=NUM_CLASSES)

    inputs = []
    flow = []

    cap = cv2.VideoCapture(video_f)
    prev = None
    while cap.isOpened():
        ret, bgr = cap.read()

        if bgr is None:
            break

        h, w, c = bgr.shape
        ratio = w / h
        w_resized = int(224 * ratio)
        resized = cv2.resize(bgr, (w_resized, 224))
        cv2.imshow("resized", resized)
        # cv2.waitKey(0)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        w1 = int((w_resized / 2) - 224 / 2)
        rgb = rgb[0:224, w1:w1 + 224, :]

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if prev is None:
            prev = gray

        curr_flow = calc_flow(gray, prev)
        flow.append(curr_flow)
        prev = gray

        cv2.imshow("cropped", cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        rgb = rgb / 127.5 - 1
        inputs.append(rgb)

        if len(inputs) == NUM_FRAMES:
            x_rgb = np.array([inputs])
            rgb_logits = rgb_model.predict(x_rgb)
            # sample_logits = rgb_logits

            x_flow = np.array([flow])
            flow_logits = flow_model.predict(x_flow)
            # sample_logits = flow_logits

            sample_logits = rgb_logits + flow_logits

            sample_logits = sample_logits[0]
            sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))

            sorted_indices = np.argsort(sample_predictions)[::-1]

            print('\nNorm of logits: %f' % np.linalg.norm(sample_logits))
            print('\nTop classes and probabilities')
            for index in sorted_indices[:20]:
                print(sample_predictions[index], sample_logits[index], kinetics_classes[index])

            i = sorted_indices[0]
            cv2.imshow("%s" % kinetics_classes[i], bgr)
            cv2.waitKey(25)
            inputs = []
            flow = []


def main(eval_type, ):
    # load the kinetics classes
    kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]

    if eval_type in ['rgb', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for RGB data
            # and load pretrained weights (trained on kinetics dataset only) 
            rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for RGB data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            rgb_model = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)

        # load RGB sample (just one example)
        rgb_sample = np.load(SAMPLE_DATA_PATH['rgb'])

        # make prediction
        rgb_logits = rgb_model.predict(rgb_sample)

    if eval_type in ['flow', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for optical flow data
            # and load pretrained weights (trained on kinetics dataset only)
            flow_model = Inception_Inflated3d(
                include_top=True,
                weights='flow_kinetics_only',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for optical flow data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            flow_model = Inception_Inflated3d(
                include_top=True,
                weights='flow_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)

        # load flow sample (just one example)
        flow_sample = np.load(SAMPLE_DATA_PATH['flow'])

        # make prediction
        flow_logits = flow_model.predict(flow_sample)

    # produce final model logits
    if eval_type == 'rgb':
        sample_logits = rgb_logits
    elif eval_type == 'flow':
        sample_logits = flow_logits
    else:  # joint
        sample_logits = rgb_logits + flow_logits

    # produce softmax output from model logit for class probabilities
    sample_logits = sample_logits[0]  # we are dealing with just one example
    sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))

    sorted_indices = np.argsort(sample_predictions)[::-1]

    print('\nNorm of logits: %f' % np.linalg.norm(sample_logits))
    print('\nTop classes and probabilities')
    for index in sorted_indices[:20]:
        print(sample_predictions[index], sample_logits[index], kinetics_classes[index])

    return


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file', type=str, help="path to the video file")
    # parser.add_argument('--eval-type',
    #                     help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).',
    #                     type=str, choices=['rgb', 'flow', 'joint'], default='joint')
    # 
    # parser.add_argument('--no-imagenet-pretrained',
    #                     help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
    #                     action='store_true')

    args = parser.parse_args()
    # m2("/Volumes/SD128/testvideo/smoke_scene_in_the_movies.mp4")
    # for fn, y in m2("/Volumes/SD128/testvideo/basic_inst/basic.mp4"):
    with open('out.jsonl', 'w') as _f:
        for fn_y in m2(args.video_file, show=True):
            s = json.dumps(fn_y)
            print(s)
            _f.write("%s\n" % s)
    # main(parser.eval_type)
