import cv2
import numpy
from keras import Input
from keras.applications import mobilenet
from keras.models import load_model
from keras.utils import CustomObjectScope

from dataset import SmokeGifSequence
from utils import yield_frames


def yield_predictions(m, video_file):
    input_shape = (299, 299, 3)
    input_image = Input((299, 299, 3))
    input_flow = Input((299, 299, 20))

    x_flow = numpy.zeros((299, 299, 20))
    old_gray = None
    flow_fn = 0
    # video_file = "/blender/storage/datasets/vg_smoke/valid/basic_inst/basic-cropped.mp4"
    # video_file = "/Volumes/SD128/getty_1525701638930.mp4"
    # video_file = "/Volumes/bstorage/datasets/vg_smoke/smoking_videos/mp4/American_History_X_smoke_h_nm_np1_fr_goo_26.avi.mp4"
    # video_file = "/Volumes/bstorage/datasets/vg_smoke/smoking_videos/mp4/smoking_again_smoke_h_cm_np1_le_goo_1.avi.mp4"
    # video_file = "/Volumes/bstorage/datasets/vg_smoke/smoking_videos/mp4/The_Matrix_4_smoke_h_nm_np1_fr_goo_1.avi.mp4"
    for fn, bgr in yield_frames(video_file):
        resized_bgr = cv2.resize(bgr, (299, 299))
        gray = cv2.cvtColor(resized_bgr, cv2.COLOR_RGB2GRAY)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        x_rgb = rgb / 127.5 - 1

        if old_gray is None:
            old_gray = gray

        if fn % 2:
            if flow_fn > 19:
                flow_fn = flow_fn - 2

                x_flow_new = numpy.zeros_like(x_flow)
                x_flow_new[:, :, :-2] = x_flow[:, :, 2:]  # shift oflow left by two positions
                x_flow = x_flow_new

            flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            x_flow[:, :, flow_fn] = mag
            x_flow[:, :, flow_fn + 1] = ang
            old_gray = gray

            flow_fn = flow_fn + 2

        # xx = [x_rgb, x_flow]
        xx = [numpy.array([x_rgb]), numpy.array([x_flow])]
        y_batch = m.predict(xx, batch_size=1)
        for y in y_batch:
            yield (fn, y)
            show = False
            if show:
                cls_id = numpy.argmax(y)
                print("fn=%d; flow_frame=%d cls_id=%d y=%s" % (fn, flow_fn, cls_id, y.round(2)))
                cv2.imshow("%d" % cls_id, resized_bgr)

                hsv = numpy.zeros_like(resized_bgr)
                mag = x_flow[:, :, 18]
                ang = x_flow[:, :, 19]
                hsv = SmokeGifSequence.flow_to_hsv(dst_hsv=hsv, mag_ang=(mag, ang))
                flow_mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imshow("mask", flow_mask)
                cv2.waitKey(25)


if __name__ == '__main__':
    # hdf = "fusion_vg_smoke_v2.h5"
    # hdf = "/blender/storage/home/chexov/smoke_attn/fusion_vg_smoke_v1.h5"
    # hdf = "/blender/storage/home/chexov/smoke_attn/fusion_vg_smoke_v2.h5"
    # hdf = "/blender/storage/home/chexov/smoke_attn/fusion_vg_smoke_v3.h5"
    hdf = "/blender/storage/home/chexov/smoke_attn/fusion_vg_smoke_v3.1.h5"
    with CustomObjectScope({'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
        m = load_model(hdf)

    video_file = "/Volumes/SD128/testvideo/smoke_scene_in_the_movies.mp4"
    for fn, y in yield_predictions(m, video_file):
        print(fn, y)
