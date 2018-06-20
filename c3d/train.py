# -*- coding: utf-8 -*-
"""C3D model for Keras

# Reference:

- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)

Based on code from @albertomontesg
"""
import os
import subprocess

import keras.backend as K
from keras import Model, Input
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import MaxPooling3D, ZeroPadding3D, Flatten, Reshape, Conv2D
from keras.layers.convolutional import Conv3D
from keras.layers.core import Dense, Dropout
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.utils.data_utils import get_file

from i3d_dataset import I3DFusionSequence

WEIGHTS_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'


def C3D_fe(input: Input, weights='sports1M'):
    """Instantiates a C3D Kerasl model

    Keyword arguments:
    weights -- weights to load into model. (default is sports1M)

    Returns:
    A Keras model.

    """

    if weights not in {'sports1M', None}:
        raise ValueError('weights should be either be sports1M or None')

    if K.image_data_format() == 'channels_last':
        shape = (16, 112, 112, 3)
    else:
        shape = (3, 16, 112, 112)

    # input = Input(shape)

    hidden = Conv3D(64, 3, activation='relu', padding='same', name='conv1')(input)
    hidden = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same', name='pool1')(hidden)

    hidden = Conv3D(128, 3, activation='relu', padding='same', name='conv2')(hidden)
    hidden = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(hidden)

    hidden = Conv3D(256, 3, activation='relu', padding='same', name='conv3a')(hidden)
    hidden = Conv3D(256, 3, activation='relu', padding='same', name='conv3b')(hidden)
    hidden = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(hidden)

    hidden = Conv3D(512, 3, activation='relu', padding='same', name='conv4a')(hidden)
    hidden = Conv3D(512, 3, activation='relu', padding='same', name='conv4b')(hidden)
    hidden = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(hidden)

    hidden = Conv3D(512, 3, activation='relu', padding='same', name='conv5a')(hidden)
    hidden = Conv3D(512, 3, activation='relu', padding='same', name='conv5b')(hidden)
    hidden = ZeroPadding3D(padding=(0, 1, 1))(hidden)
    hidden = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(hidden)

    model = Model(inputs=input, outputs=hidden)
    if weights == 'sports1M':
        weights_path = get_file('sports1M_weights_tf.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='b7a93b2f9156ccbebe3ca24b41fc5402')

        model.load_weights(weights_path, by_name=True)

    return model


if __name__ == '__main__':
    NUM_FRAMES = 16
    input_c3d = Input((NUM_FRAMES, 112, 112, 3))
    fe = C3D_fe(input_c3d, weights='sports1M')
    for l in fe.layers:
        if "conv5a" == l.name:
            break
        l.trainable = False

    classifier = Flatten()(fe.get_output_at(0))
    classifier = Reshape((4, 4, 512))(classifier)

    classifier = Dropout(0.5)(classifier)
    classifier = Conv2D(64, 1, padding="same", activation="relu", name="conv2d_cls1")(classifier)

    classifier = Dropout(0.5)(classifier)
    classifier = Conv2D(32, 1, padding="same", activation="relu", name="conv2d_cls2")(classifier)

    classifier = Flatten()(classifier)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(2, activation='softmax', name='fc9')(classifier)

    model = Model(inputs=input_c3d, outputs=classifier)

    # plot_model(model)
    model.compile(Adam(lr=1e-4), binary_crossentropy, ["accuracy", ])
    model.summary()

    data_dir = "/blender/storage/datasets/vg_smoke/"
    train_seq = I3DFusionSequence(data_dir, "train.txt",
                                  input_hw=(112, 112),
                                  batch_size=12, num_frames_in_sequence=NUM_FRAMES,
                                  only_spacial=True)
    val_seq = I3DFusionSequence(data_dir, "validate.txt",
                                input_hw=(112, 112),
                                batch_size=12, num_frames_in_sequence=NUM_FRAMES,
                                only_spacial=True)

    hdf = "c3d_v2.hdf"
    assert subprocess.call("git tag %s" % hdf, shell=True) == 0, "rename the experiment or delete git tag"

    log_dir = os.path.join("./logs", os.path.basename(hdf))
    model.fit_generator(train_seq, len(train_seq), epochs=10,
                        use_multiprocessing=True, workers=10,
                        validation_data=val_seq, validation_steps=len(val_seq),
                        verbose=1,
                        callbacks=[TensorBoard(log_dir), ModelCheckpoint(hdf, verbose=1, save_best_only=True)],
                        )
