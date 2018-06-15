import numpy as np
from keras import Model, Input
from keras.applications.mobilenet import MobileNet
from keras.engine import Layer
from keras.layers import Conv3D, ZeroPadding3D, BatchNormalization, Activation, DepthwiseConv2D
from copy import deepcopy


def conv2d3d(layer: Layer):
    if type(layer).__name__ != 'Conv2D':
        raise TypeError("not conv2d layer")
    config = layer.get_config()

    ks = config['kernel_size']
    config['kernel_size'] = (ks[0],) + ks

    st = config['strides']
    config['strides'] = (st[0],) + st

    dl = config['dilation_rate']
    config['dilation_rate'] = (dl[0],) + dl

    result = Conv3D(**config)

    return result


def input2d3d(layer: Layer, t_filters):
    config = layer.get_config()

    result = Input()


def inflater(model_2d: Model, t_channels):
    l3_layers = {}
    for l2 in model_2d.layers:
        layer_type = type(l2).__name__
        name = l2.name
        if layer_type == 'InputLayer':
            shape = (t_channels,) + l2.input_shape[1:]
            l3 = Input(shape, name=name)
        elif type(l2).__name__ == 'Conv2D':
            l3 = conv2d3d(l2)
        else:
            l3 = deepcopy(l2)
        l3_layers[name] = l3

    inputs = []
    outputs = []

    model_3d = Model(inputs=inputs, outputs=outputs)
    return model_3d


def transfer_weights(model_2d: Model, model_3d: Model):
    for l2 in model_2d.layers:
        if type(l2).__name__ == 'Conv2D':
            n2 = l2.name
            try:
                l3 = model_3d.get_layer(n2)
                s = l3.kernel.shape[0]
                w = l2.get_weights()
                w[0] = np.array(w[0] * s)
                l3.set_weights(w)
            except ValueError:
                pass


def mobilenet_3d(input_shape: tuple, model_2d: Model):
    input_image = Input(input_shape)
    x = ZeroPadding3D()(input_image)
    x = conv2d3d(model_2d.get_layer('conv1'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding3D()(x)
    x = DepthwiseConv2D()(x)


def main():
    # inp = Input((41, 42, 11))
    # x = Conv2D(8, (7, 9))(inp)
    # model = Model(inputs=inp, outputs=x)

    model_2d = MobileNet(input_shape=(224, 224, 3), include_top=False)

    model_3d = inflater(model_2d, 5)
    transfer_weights(model_2d, model_3d)

    # inp = Input((41, 42, 43, 15))
    # input2d3d(inp, 5)
    # Conv2D3D(model.layers[1], 5, inp)
    # plot_model(model=model, show_shapes=True)
    # model.summary()
    # inflater(model)


if __name__ == '__main__':
    main()
