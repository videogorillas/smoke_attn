import os

from keras.callbacks import TensorBoard, ModelCheckpoint

from dataset import SmokeGifSequence
from model import inceptionv3_m1

if __name__ == '__main__':
    input_shape = (299, 299, 3)

    hdf = "vg_smoke_spacial_v1.h5"
    m = inceptionv3_m1(class_count=2, image_shape_hwc=input_shape)

    # m.load_weights(hdf)
    # plot_model(m, show_shapes=True)
    # load_model(hdf)
    m.summary()

    data_dir = "/blender/storage/datasets/vg_smoke"
    train_seq = SmokeGifSequence(data_dir, neg_txt='negatives.txt', pos_txt='positives.txt',
                                 input_shape_hwc=input_shape,
                                 only_spacial=True)

    val_seq = SmokeGifSequence(data_dir, neg_txt='validate.txt', pos_txt='validate.txt',
                               input_shape_hwc=input_shape,
                               batch_size=3,
                               only_spacial=True)

    log_dir = os.path.join("./logs", os.path.basename(hdf))

    m.fit_generator(train_seq, len(train_seq), epochs=10,
                    use_multiprocessing=True, workers=5,
                    validation_data=val_seq, validation_steps=42,
                    verbose=1, callbacks=[TensorBoard(log_dir=log_dir), ModelCheckpoint(hdf)],
                    )
