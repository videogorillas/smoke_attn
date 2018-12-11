#!/usr/bin/env python
import itertools
from multiprocessing import Queue

import cv2
import image2pipe
import numpy
from keras.activations import linear
from keras.models import load_model
from vis.utils.utils import find_layer_idx, apply_modifications
from vis.visualization import visualize_cam, overlay


def main():
    model = load_model("/blender/storage/vgmodels/smoking/inception_v3.h5")
    model.summary()
    layer_idx = find_layer_idx(model, 'dense_1')
    model.layers[layer_idx].activation = linear
    model = apply_modifications(model)

    show = True
    penultimater_layer_idx = find_layer_idx(model, 'max_pooling2d_4')

    # bgr = cv2.imread("/Volumes/SD128/smoke.png", 1)
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cap = cv2.VideoCapture("/Volumes/SD128/testvideo/basic_inst/basic.mp4")
    fps = str(cap.get(cv2.CAP_PROP_FPS))

    q = Queue(maxsize=42)
    encoder = image2pipe.StitchVideoProcess(q, "out.mp4", fps, (300, 300), "rgb24", muxer="mov")
    encoder.daemon = True
    encoder.start()

    fn_iter = itertools.count()
    while cap.isOpened():
        fn = next(fn_iter)
        print(fn)
        ret, bgr = cap.read()
        if bgr is None:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xrgb = cv2.resize(rgb, (299, 299))
        x = xrgb / 127.5 - 1

        layer_idx = 312
        heatmap = visualize_cam(model, layer_idx, filter_indices=0,  # 20 for ouzel and 292 for tiger 
                                seed_input=x, backprop_modifier=None,  # relu and guided don't work
                                penultimate_layer_idx=310  # 310 is concatenation before global average pooling
                                )
        jetmap = overlay(heatmap, xrgb, 0.4)

        if show:
            cv2.imshow("activations", cv2.cvtColor(numpy.uint8(jetmap), cv2.COLOR_RGB2BGR))
            cv2.waitKey(25)

        jetmap = cv2.resize(jetmap, (300, 300))
        q.put((fn, jetmap))

        # plt.subplot(212)
        # plt.imshow(xrgb)
        # plt.subplot(221)
        # plt.imshow(jetmap)
        # plt.show()

    cap.release()
    q.put(None)
    encoder.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
