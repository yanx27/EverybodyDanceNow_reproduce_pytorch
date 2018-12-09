import numpy as np
import cv2

def main():
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        is_capturing, _ = vc.read()
    else:
        is_capturing = False

    cnt = 0
    while is_capturing:
        is_capturing, img = vc.read()
        # img = img[:, 80:80+480]
        cv2.imwrite('../../data/target/images/img_%d.png'%cnt, img)
        cv2.imshow('video', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
        cnt += 1
    vc.release()
    cv2.destroyAllWindows()

main()
