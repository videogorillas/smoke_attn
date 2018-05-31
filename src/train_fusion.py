import os

from keras import Input, Model
from keras.applications import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.backend import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv2D, concatenate, Dense, Flatten

from dataset import SmokeGifSequence

if __name__ == '__main__':
    input_shape = (299, 299, 3)
    # hdf = "m1.h5"
    hdf = "fusion_vg_smoke_v1.h5"

    input_image = Input((299, 299, 3))
    input_flow = Input((299, 299, 20))

    fe = InceptionV3(include_top=False, input_tensor=input_image)
    for l in fe.layers:
        l.trainable = False
    x_rgb = fe.get_output_at(0)
    x_rgb = Conv2D(8, 3, padding='same', activation='relu')(x_rgb)

    m = MobileNet(include_top=False, input_tensor=input_flow, weights=None)
    x_flow = m.get_output_at(0)
    x_flow = Conv2D(8, 3, padding='valid', activation='relu')(x_flow)
    x = concatenate([x_rgb, x_flow])
    x = Flatten()(x)
    x = Dense(2, activation="softmax")(x)

    m = Model(inputs=[input_image, input_flow], outputs=x)

    # load_model(hdf)
    # m.load_weights(hdf)
    m.compile("adam", categorical_crossentropy, metrics=["accuracy"])
    # plot_model(m, show_shapes=True)

    m.summary()

    # data_dir = "/blender/storage/datasets/smoking/gifs/"
    data_dir = "/blender/storage/datasets/vg_smoke"

    train_seq = SmokeGifSequence(data_dir, neg_txt='negatives.txt', pos_txt='positives.txt',
                                 input_shape_hwc=input_shape)
    # val_seq = SmokeGifSequence(data_dir, neg_txt='validate_neg.txt', pos_txt='validate_pos.txt',
    #                            input_shape_hwc=input_shape,
    #                            only_temporal=True)
    # 
    log_dir = os.path.join("./logs", os.path.basename(hdf))

    m.fit_generator(train_seq, len(train_seq), epochs=10,
                    use_multiprocessing=True, workers=5,
                    # validation_data=val_seq, validation_steps=len(val_seq),
                    verbose=1, callbacks=[TensorBoard(log_dir), ModelCheckpoint(hdf)],
                    )
