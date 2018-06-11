import os

from keras import Model
from keras.applications import InceptionV3
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense
from keras.optimizers import SGD

from dataset import SmokeGifSequence

if __name__ == '__main__':
    input_shape = (299, 299, 3)

    hdf = "vg_smoke_spacial_v1.1.h5"
    feature_extractor = InceptionV3(input_shape=input_shape, include_top=False, pooling='avg')
    for l in feature_extractor.layers[:]:
        # if "mixed9" == l.name:
        #     break

        l.trainable = False

    x = Dense(2, activation='softmax')(feature_extractor.output)
    m = Model(inputs=feature_extractor.input, outputs=x)

    optimizer = SGD(1e-4, decay=1e-6, nesterov=True, momentum=0.9)
    m.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])

    m.load_weights(hdf)
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

    m.fit_generator(train_seq, len(train_seq), epochs=20,
                    use_multiprocessing=True, workers=5,
                    validation_data=val_seq, validation_steps=42,
                    verbose=1, callbacks=[TensorBoard(log_dir=log_dir), ModelCheckpoint(hdf, save_best_only=True)],
                    )
