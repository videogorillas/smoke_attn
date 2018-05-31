import os

from keras.callbacks import TensorBoard, ModelCheckpoint

from dataset import SmokeGifSequence
from model import inceptionv3_m1

if __name__ == '__main__':
    input_shape = (299, 299, 3)
    # hdf = "m1.h5"

    # hdf = "/blender/storage/home/chexov/smoke_attn/m1.h5"
    # m = inceptionv3_m1(class_count=2, image_shape_hwc=input_shape)
    # m.load_weights(hdf)
    # # plot_model(m, show_shapes=True)
    # # load_model(hdf)
    # m.summary()

    # data_dir = "/blender/storage/datasets/smoking/gifs/"

    # train_seq = SmokeGifSequence(data_dir, neg_txt='train_neg.txt', pos_txt='train_pos.txt',
    #                              input_shape_hwc=input_shape)
    # val_seq = SmokeGifSequence(data_dir, neg_txt='validate_neg.txt', pos_txt='validate_pos.txt',
    #                            input_shape_hwc=input_shape)
    # 
    # m.fit_generator(train_seq, len(train_seq), epochs=10,
    #                 validation_data=val_seq, validation_steps=len(val_seq),
    #                 verbose=1, callbacks=[TensorBoard(), ModelCheckpoint(hdf)],
    #                 )

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
    # val_seq = SmokeGifSequence(data_dir, neg_txt='validate_neg.txt', pos_txt='validate_pos.txt',
    #                            input_shape_hwc=input_shape,
    #                            only_temporal=True)

    log_dir = os.path.join("./logs", os.path.basename(hdf))
    m.fit_generator(train_seq, len(train_seq), epochs=10,
                    use_multiprocessing=True, workers=5,
                    # validation_data=val_seq, validation_steps=len(val_seq),
                    verbose=1, callbacks=[TensorBoard(log_dir=log_dir), ModelCheckpoint(hdf)],
                    )
