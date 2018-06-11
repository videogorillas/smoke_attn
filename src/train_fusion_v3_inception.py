import os

from keras import Input, Model
from keras.activations import relu
from keras.applications.inception_v3 import InceptionV3
from keras.backend import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, concatenate
from keras.optimizers import Adam

from dataset import SmokeGifSequence

if __name__ == '__main__':
    input_shape = (299, 299, 3)

    input_image = Input((299, 299, 3))
    input_flow = Input((299, 299, 20))

    fe = InceptionV3(include_top=False, input_tensor=input_image, weights='imagenet', pooling='avg')
    for l in fe.layers:
        l.trainable = False
    x_rgb = fe.get_output_at(0)

    m_flow = InceptionV3(include_top=False, input_tensor=input_flow, weights=None, pooling='avg')
    for l in m_flow.layers:
        l.trainable = True
        l.name = "flow-" + l.name

    x_flow = m_flow.get_output_at(0)

    x = concatenate([x_rgb, x_flow])
    x = Dense(1024, activation=relu)(x)
    x = Dense(2, activation="softmax")(x)
    m = Model(inputs=[input_image, input_flow], outputs=x)

    # load_model(hdf)
    hdf = "fusion_vg_smoke_inception_v3.4.h5"
    # m.load_weights(hdf)
    m.compile(Adam(lr=1e-4), categorical_crossentropy, metrics=["accuracy"])
    # plot_model(m, show_shapes=True)

    m.summary()

    # data_dir = "/blender/storage/datasets/smoking/gifs/"
    data_dir = "/blender/storage/datasets/vg_smoke"
    train_seq = SmokeGifSequence(data_dir, neg_txt='negatives.txt', pos_txt='positives.txt',
                                 input_shape_hwc=input_shape)

    val_seq = SmokeGifSequence(data_dir, neg_txt='validate.txt', pos_txt='validate.txt',
                               input_shape_hwc=input_shape,
                               batch_size=2,
                               only_spacial=False, only_temporal=False)

    log_dir = os.path.join("./logs", os.path.basename(hdf))

    m.fit_generator(train_seq, len(train_seq), epochs=10,
                    use_multiprocessing=True, workers=10,
                    validation_data=val_seq, validation_steps=42,
                    verbose=1, callbacks=[TensorBoard(log_dir), ModelCheckpoint(hdf, save_best_only=True)],
                    )
