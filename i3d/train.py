import os

from keras import Model, Input
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, concatenate, Flatten
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from i3d_dataset import I3DFusionSequence
from i3d_inception import Inception_Inflated3d

if __name__ == '__main__':
    NUM_FRAMES = 16
    FRAME_HEIGHT = 224
    FRAME_WIDTH = 224
    NUM_RGB_CHANNELS = 3
    NUM_FLOW_CHANNELS = 2
    NUM_CLASSES = 2

    rgb_input = Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS))
    rgb_model = Inception_Inflated3d(
        include_top=False,
        # weights='rgb_imagenet_and_kinetics',
        weights='rgb_kinetics_only',
        input_tensor=rgb_input,
        classes=NUM_CLASSES)
    for l in rgb_model.layers:
        l.trainable = False
    rgb_y = rgb_model.get_output_at(0)

    flow_input = Input(shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS))
    flow_model = Inception_Inflated3d(
        include_top=False,
        # weights='flow_imagenet_and_kinetics',
        weights='flow_kinetics_only',
        input_tensor=flow_input,
        classes=NUM_CLASSES)
    for l in flow_model.layers:
        l.trainable = False
        l.name = "flow-" + l.name
    flow_y = flow_model.get_output_at(0)

    y = concatenate([rgb_y, flow_y])
    y = Flatten()(y)
    y = Dense(2, activation="softmax", name="fc9")(y)
    model = Model(inputs=[rgb_input, flow_input], outputs=y)

    # plot_model(model)
    model.compile(Adam(lr=1e-4), binary_crossentropy, metrics=["accuracy", ])
    model.summary()

    data_dir = "/blender/storage/datasets/vg_smoke/"
    train_seq = I3DFusionSequence(data_dir, "train.txt", batch_size=16, num_frames=16)
    val_seq = I3DFusionSequence(data_dir, "validate.txt", batch_size=16, num_frames=16)

    hdf = "i3d_kinetics_finetune_v1.0.hdf"

    log_dir = os.path.join("./logs", os.path.basename(hdf))
    model.fit_generator(train_seq, len(train_seq), epochs=10,
                        use_multiprocessing=True, workers=10,
                        validation_data=val_seq, validation_steps=len(val_seq),
                        verbose=1, callbacks=[TensorBoard(log_dir), ModelCheckpoint(hdf, save_best_only=True)],
                        )
