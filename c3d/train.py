# -*- coding: utf-8 -*-
"""C3D model for Keras

# Reference:

- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)

Based on code from @albertomontesg
"""
import os

import keras.backend as K
from keras import Model, Input
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import MaxPooling3D, ZeroPadding3D, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.core import Dense, Dropout
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.utils.data_utils import get_file

from c3d_dataset import C3DSequence

WEIGHTS_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'


def C3D_notop(input: Input, weights='sports1M'):
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

    hidden = Flatten()(hidden)

    hidden = Dense(4096, activation='relu', name='fc6')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(4096, activation='relu', name='fc7')(hidden)

    # hidden = Dropout(0.5)(hidden)
    # hidden = Dense(2, activation='softmax', name='fc9')(hidden)

    model = Model(inputs=input, outputs=hidden)
    if weights == 'sports1M':
        weights_path = get_file('sports1M_weights_tf.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='b7a93b2f9156ccbebe3ca24b41fc5402')

        model.load_weights(weights_path, by_name=True)

    return model


if __name__ == '__main__':
    input_c3d = Input((16, 112, 112, 3))
    fe = C3D_notop(input_c3d, weights='sports1M')
    for l in fe.layers:
        l.trainable = False

    x = fe.get_output_at(0)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax", name="fc9")(x)
    model = Model(inputs=[input_c3d], outputs=x)

    # plot_model(model)
    model.compile(SGD(lr=1e-4), binary_crossentropy, ["accuracy", ])
    model.summary()

    train_seq = C3DSequence("/blender/storage/datasets/vg_smoke/", "train.txt", batch_size=16)
    val_seq = C3DSequence("/blender/storage/datasets/vg_smoke/", "validate.txt", batch_size=16)

    hdf = "c3d_sports1M_v1.hdf"

    log_dir = os.path.join("./logs", os.path.basename(hdf))
    model.fit_generator(train_seq, len(train_seq), epochs=10,
                        use_multiprocessing=True, workers=10,
                        validation_data=val_seq, validation_steps=len(val_seq),
                        verbose=1, callbacks=[TensorBoard(log_dir), ModelCheckpoint(hdf, save_best_only=True)],
                        )
