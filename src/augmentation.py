#!/usr/bin/env python
# encoding: utf-8

import io
import random

import numpy as np
from PIL import Image
from PIL.ImageEnhance import Contrast
from imgaug import augmenters as iaa
from imgaug.parameters import StochasticParameter, Deterministic
from scipy.misc import imread, imrotate, imresize


class ImageAugmentation:
    def __init__(self, images_list=None):
        self.images_list = images_list
        self.p = 0.21
        self.blur = None
        self.transform = None
        self.renew()

    def renew(self):
        self.blur = self.blur_sequence(self.p)
        self.transform = self.transform_sequence(self.p)

    def blur_sequence(self, probability):
        sometimes = lambda aug: iaa.Sometimes(probability, aug)
        seq = iaa.Sequential([
            sometimes(iaa.Multiply((0.8, 1.2))),  # change brightness, doesn't affect BBs
            sometimes(iaa.GaussianBlur(sigma=1.0)),
            sometimes(iaa.SaltAndPepper(0.01)),
            sometimes(iaa.ContrastNormalization((0.5, 1.5))),
            sometimes(JPEGArtifacts(random.randrange(90, 100))),
            sometimes(ColorMix(0.1)),
            sometimes(ColorNoise(0.1)),
            sometimes(ContrastMix(0.1)),
            sometimes(iaa.Dropout(p=BinomialRows(0.9))),
            sometimes(iaa.Dropout(p=BinomialColumns(0.9)))
        ])

        if self.images_list is not None:
            seq.add(sometimes(ImageMix(self.images_list)))

        return seq

    def transform_sequence(self, probability):
        sometimes = lambda aug: iaa.Sometimes(probability, aug)
        seq = iaa.Sequential([
            sometimes(iaa.Affine(rotate=(-10, 10))),
            sometimes(iaa.Affine(scale=(0.9, 1.1))),
            sometimes(iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
            sometimes(iaa.Affine(shear=(-7, 7))),
            sometimes(iaa.Fliplr(1))
        ])
        seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start

        return seq_det


class JPEGArtifacts(iaa.Augmenter):
    def __init__(self, quality=7):
        super(JPEGArtifacts, self).__init__()
        self.quality = quality

    def get_parameters(self):
        return [self.quality]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        for i, img in enumerate(images):
            temp = io.BytesIO()
            im = Image.fromarray(img)
            # https://en.wikipedia.org/wiki/Chroma_subsampling
            im.save(temp, format="jpeg", quality=self.quality, subsampling=random.randrange(0, 2))
            temp.seek(0)
            result[i] = imread(temp)
        return result


# tint with random color
class ColorMix(iaa.Augmenter):
    def __init__(self, max_intensity: float = 0.042):
        super(ColorMix, self).__init__()
        self.max_intensity = max_intensity

    def get_parameters(self):
        return [self.max_intensity]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        for i, img in enumerate(images):
            f = random.uniform(0, self.max_intensity)
            result[i] = np.array(img * (1 - f) + f * np.random.uniform(0, 255, 3), dtype="uint8")
        return result


# add random color noise to image data
class ColorNoise(iaa.Augmenter):
    def __init__(self, max_intensity: float = 0.042):
        super(ColorNoise, self).__init__()
        self.max_intensity = max_intensity

    def get_parameters(self):
        return [self.max_intensity]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        for i, img in enumerate(images):
            f = random.uniform(0, self.max_intensity)
            result[i] = np.array(img * (1 - f) + f * np.random.uniform(0, 255, img.shape), dtype="uint8")
        return result


class ContrastMix(iaa.Augmenter):
    def __init__(self, max_contranst_changes: float = 0.042):
        super(ContrastMix, self).__init__()
        self.contrast_changes = max_contranst_changes

    def get_parameters(self):
        return [self.contrast_changes]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        for i, img in enumerate(images):
            f = random.uniform(1 - self.contrast_changes, 1 + self.contrast_changes)
            im = Image.fromarray(img)
            result[i] = np.array(Contrast(im).enhance(f), dtype="uint8")
        return result


# tint with random image from list
class ImageMix(iaa.Augmenter):
    def __init__(self, images_list: list, intensity: float = 0.042):
        super(ImageMix, self).__init__()
        self.intensity = intensity
        self.image_list = images_list

    def get_parameters(self):
        return [self.intensity, self.image_list]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        for i, img in enumerate(images):
            f = random.uniform(0, self.intensity)
            mix_image = imread(random.choice(self.image_list))
            mix_image = imrotate(mix_image, random.uniform(0, 360))
            mix_image = imresize(mix_image, img.shape)
            result[i] = np.array(img * (1 - f) + f * mix_image, dtype="uint8")
        return result


class BinomialRows(StochasticParameter):
    def __init__(self, p):
        super(BinomialRows, self).__init__()
        self.p = Deterministic(float(p))

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        h, w, c = size
        drops = random_state.binomial(1, p, (h, 1, c))
        drops_rows = np.tile(drops, (1, w, 1))
        return drops_rows


class BinomialColumns(StochasticParameter):
    def __init__(self, p):
        super(BinomialColumns, self).__init__()
        self.p = Deterministic(float(p))

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        h, w, c = size
        drops = random_state.binomial(1, p, (1, w, c))
        drops_columns = np.tile(drops, (h, 1, 1))
        return drops_columns
