import os

from keras import Input, Model
from keras.activations import relu
from keras.applications import VGG19
from keras.backend import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, concatenate
from keras.utils import plot_model

from dataset import SmokeGifSequence

if __name__ == '__main__':
    input_shape = (299, 299, 3)

    input_image = Input((299, 299, 3))
    input_flow = Input((299, 299, 20))

    # fe = InceptionV3(include_top=False, input_tensor=input_image)
    fe = VGG19(include_top=False, input_tensor=input_image)
    fe.summary()
    plot_model(fe)

    for l in fe.layers:
        l.trainable = False
    x_rgb = fe.get_output_at(0)
    # x_rgb = Conv2D(8, 3, padding='same', activation='relu')(x_rgb)

    # m = MobileNet(include_top=False, input_tensor=input_flow, weights=None, pooling='avg')
    # x_flow = m.get_output_at(0)
    x_flow = Conv2D(8, 3, strides=2, activation='relu')(input_flow)
    x_flow = Conv2D(8, 3, strides=2, activation='relu')(x_flow)
    x_flow = Conv2D(16, 3, strides=2, activation='relu')(x_flow)
    x_flow = Conv2D(16, 3, strides=2, activation='relu')(x_flow)
    x_flow = Conv2D(32, 3, strides=2, activation='relu')(x_flow)
    # x_flow = Flatten()(x_flow)

    x = concatenate([x_rgb, x_flow])
    x = Conv2D(64, 3, activation=relu)(x)
 
    x = Flatten()(x)
    # x = Dense(1024, activation=relu)(x)

    x = Dense(2, activation="softmax")(x)
    m = Model(inputs=[input_image, input_flow], outputs=x)

    # load_model(hdf)
    hdf = "fusion_vg_smoke_v3.1.h5"
    # m.load_weights(hdf)
    m.compile("adam", categorical_crossentropy, metrics=["accuracy"])
    # plot_model(m, show_shapes=True)

    m.summary()

    # data_dir = "/blender/storage/datasets/smoking/gifs/"
    data_dir = "/blender/storage/datasets/vg_smoke"

    train_seq = SmokeGifSequence(data_dir, neg_txt='negatives.txt', pos_txt='positives.txt',
                                 input_shape_hwc=input_shape)
    valid_data_dir = "/blender/storage/datasets/vg_smoke/valid/"
    val_seq = SmokeGifSequence(valid_data_dir, neg_txt='validate.txt', pos_txt='validate.txt',
                               input_shape_hwc=input_shape,
                               batch_size=2,
                               only_spacial=False, only_temporal=False)

    log_dir = os.path.join("./logs", os.path.basename(hdf))

    m.fit_generator(train_seq, len(train_seq), epochs=20,
                    use_multiprocessing=True, workers=10,
                    validation_data=val_seq, validation_steps=42,
                    verbose=1, callbacks=[TensorBoard(log_dir), ModelCheckpoint(hdf, save_best_only=True)],
                    )
