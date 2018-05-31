import cv2
import numpy as np
from keras.models import load_model

from utils import yield_frames


def main():
    model = load_model('model.h5')
    video_file = 'basic.mp4'

    # drop_every_n_frame = 2

    h, w = model.get_input_shape_at(0)[1:3]
    flows_count = model.get_input_shape_at(1)[-1]

    frames = [bgr for fn, bgr in yield_frames(video_file)]
    frame_count = len(frames)

    resizes = [cv2.resize(bgr, dsize=(w, h)) for bgr in frames]
    grays = [cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) for bgr in resizes]
    rgbs = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in resizes]

    flows = []
    mags = []
    angs = []

    for i, gray in enumerate(grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(grays[i - 1], gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flows.append(flow)
        mags.append(mag)
        angs.append(ang)

    batch_size = 1

    for i in range(0, frame_count-flows_count, 5):
        xrgb_batch = rgbs[i + flows_count]
        xflow_batch = np.zeros(shape=(batch_size, h, w, flows_count * 2))
        batch_no = 0
        for j in range(flows_count):
            xflow_batch[batch_no, , :, 2 * j] = mags[i + j]
            xflow_batch[batch_no, :, 2 * j + 1] = angs[i + j]
        predictions = model.predict([xrgb_batch, xflow_batch])


if __name__ == '__main__':
    main()
