import cv2

if __name__ == '__main__':
    with open('positives.txt', 'r') as _f:
        for fn in _f.readlines():
            fn = fn.strip()
            cap = cv2.VideoCapture(fn)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(fn, frames)
