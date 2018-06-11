import os

from keras import Input, Model
from keras.activations import relu
from keras.applications import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.backend import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv2D, concatenate, Dense, Flatten
from keras.optimizers import SGD, Adam

from dataset import SmokeGifSequence

if __name__ == '__main__':
    input_shape = (299, 299, 3)

    input_image = Input((299, 299, 3))
    input_flow = Input((299, 299, 20))

    fe = InceptionV3(include_top=False, input_tensor=input_image)
    for l in fe.layers:
        # if "mixed9" == l.name:
        #     break

        l.trainable = False

    x_rgb = fe.get_output_at(0)
    x_rgb = Conv2D(8, 3, padding='same', activation='relu')(x_rgb)

    m = MobileNet(include_top=False, input_tensor=input_flow, weights=None)
    x_flow = m.get_output_at(0)
    x_flow = Conv2D(8, 3, padding='valid', activation='relu')(x_flow)

    x = concatenate([x_rgb, x_flow])
    x = Conv2D(64, 3, activation=relu)(x)
    x = Flatten()(x)

    x = Dense(2, activation="softmax")(x)

    m = Model(inputs=[input_image, input_flow], outputs=x)

    hdf = "fusion_vg_smoke_v1.h5"
    m.load_weights(hdf)

    # m.compile(SGD(lr=1e-4), categorical_crossentropy, metrics=["accuracy"])
    m.compile(Adam(), categorical_crossentropy, metrics=["accuracy"])
    # plot_model(m, show_shapes=True)

    m.summary()

    # data_dir = "/blender/storage/datasets/smoking/gifs/"
    data_dir = "/blender/storage/datasets/vg_smoke"

    train_seq = SmokeGifSequence(data_dir, neg_txt='negatives.txt', pos_txt='positives.txt',
                                 input_shape_hwc=input_shape)
    log_dir = os.path.join("./logs", os.path.basename(hdf))

    val_seq = SmokeGifSequence(data_dir, neg_txt='validate.txt', pos_txt='validate.txt',
                               input_shape_hwc=input_shape,
                               batch_size=2,
                               only_spacial=False, only_temporal=False)

    m.fit_generator(train_seq, len(train_seq), epochs=10,
                    use_multiprocessing=True, workers=5,
                    validation_data=val_seq, validation_steps=42,
                    verbose=1, callbacks=[TensorBoard(log_dir), ModelCheckpoint(hdf)],
                    )
