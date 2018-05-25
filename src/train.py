from keras.callbacks import TensorBoard, ModelCheckpoint

from src.dataset import SmokeGifSequence, BatchSeq
from src.model import inceptionv3_m1

if __name__ == '__main__':
    input_shape = (299, 299, 3)
    m = inceptionv3_m1(class_count=2, image_shape_hwc=input_shape)
    m.summary()

    data_dir = "/bstorage/datasets/smoking/gifs/"

    train_seq = SmokeGifSequence(data_dir, neg_txt='train_neg.txt', pos_txt='train_pos.txt',
                                 input_shape_hwc=input_shape)
    validate_seq = SmokeGifSequence(data_dir, neg_txt='test_neg.txt', pos_txt='validate_pos.txt',
                                    input_shape_hwc=input_shape)

    train = BatchSeq(train_seq, batch_size=16)
    val = BatchSeq(validate_seq, batch_size=16)

    m.fit_generator(train, len(train), epochs=10,
                    validation_data=val, validation_steps=len(val),
                    verbose=1, callbacks=[TensorBoard(), ModelCheckpoint("m1.h5")],
                    )
