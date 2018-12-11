import cv2
import numpy

if __name__ == '__main__':
    cap = cv2.VideoCapture()
    cap.open("/Volumes/SD128/macg/BX138_SRNA_03.mov")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 400)

    TVL1 = cv2.DualTVL1OpticalFlow_create()
    prev_gray = None
    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        bgr = cv2.resize(bgr, (480, 480))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray

        flow = TVL1.calc(prev_gray, gray, None)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        flow_hsv = numpy.zeros_like(bgr)
        flow_hsv[..., 0] = ang * 180 / numpy.pi / 2
        flow_hsv[..., 1] = 255
        flow_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        cv2.imshow("flow", cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR))
        cv2.imshow("frame", bgr)
        cv2.waitKey(25)

    cap.release()
