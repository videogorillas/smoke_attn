from keras import Model
from keras.applications import InceptionV3
from keras.layers import Dense
from keras.optimizers import SGD


def inceptionv3_m1(class_count: int, image_shape_hwc: tuple):
    optimizer = SGD(1e-4, nesterov=True, momentum=0.9)

    feature_extractor = InceptionV3(input_shape=image_shape_hwc, include_top=False, pooling='avg')
    for l in feature_extractor.layers[:]:
        # if "mixed9" == l.name:
        #     break

        l.trainable = False

    x = Dense(class_count, activation='softmax')(feature_extractor.output)
    m = Model(inputs=feature_extractor.input, outputs=x)
    m.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
    return m
