'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''

import argparse
from cv2 import DualTVL1OpticalFlow_create as DualTVL1

import cv2
import numpy as np

from i3d_inception import Inception_Inflated3d

NUM_FRAMES = 25
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


def calc_flow(gray, prev):
    curr_flow = TVL1.calc(prev, gray, None)
    curr_flow[curr_flow >= 20] = 20
    curr_flow[curr_flow <= -20] = -20
    # scale to [-1, 1]
    max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
    curr_flow = curr_flow / max_val(curr_flow)
    return curr_flow


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
    parser.add_argument('--eval-type',
                        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).',
                        type=str, choices=['rgb', 'flow', 'joint'], default='joint')

    parser.add_argument('--no-imagenet-pretrained',
                        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
                        action='store_true')

    args = parser.parse_args()
    m("/Volumes/SD128/testvideo/smoke_scene_in_the_movies.mp4")
    # m("/Volumes/SD128/testvideo/basic_inst/basic.mp4")
    # main(parser.eval_type)
