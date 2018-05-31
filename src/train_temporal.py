import os

from keras.applications import MobileNet
from keras.backend import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint

from dataset import SmokeGifSequence

if __name__ == '__main__':
    input_shape = (299, 299, 3)
    # hdf = "m1.h5"
    hdf = "temporal_vg_smoke_v1.h5"

    m = MobileNet(input_shape=(299, 299, 20), weights=None, classes=2)
    # load_model(hdf)
    m.load_weights(hdf)
    m.compile("adam", categorical_crossentropy, metrics=["accuracy"])
    # plot_model(m, show_shapes=True)

    m.summary()

    # data_dir = "/blender/storage/datasets/smoking/gifs/"
    data_dir = "/blender/storage/datasets/vg_smoke"

    train_seq = SmokeGifSequence(data_dir, neg_txt='negatives.txt', pos_txt='positives.txt',
                                 input_shape_hwc=input_shape,
                                 only_temporal=True)
    # val_seq = SmokeGifSequence(data_dir, neg_txt='validate_neg.txt', pos_txt='validate_pos.txt',
    #                            input_shape_hwc=input_shape,
    #                            only_temporal=True)

    log_dir = os.path.join("./logs", os.path.basename(hdf))
    m.fit_generator(train_seq, len(train_seq), epochs=10,
                    use_multiprocessing=True, workers=5,
                    # validation_data=val_seq, validation_steps=len(val_seq),
                    verbose=1, callbacks=[TensorBoard(log_dir), ModelCheckpoint(hdf)],
                    )
