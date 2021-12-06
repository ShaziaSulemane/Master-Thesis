#!/usr/bin/python3
import math

import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

PATH = r'/home/ssulemane/Documents/YOLO/maps'
angles = []
medians = []

for imgdir in os.listdir(r'/home/ssulemane/Documents/YOLO/maps'):
    if not (os.path.isdir(os.path.join(PATH, imgdir))):

        img_plt = plt.imread(os.path.join(PATH, imgdir))
        img = cv.imread(os.path.join(PATH, imgdir))

        if np.allclose(img_plt[:, :, 3], 1):
            print(os.path.join(PATH, imgdir))

            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_bin = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY)
            lines = cv.HoughLinesP(np.float32(img_bin), 1, math.pi/180.0, 100, minLineLength=100, maxLineGap=5)

            for [[x1, y1, x2, y2]] in lines:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)

            medians.append(np.median(angles))

final_median = np.median(medians)
print(f'Final Median: {final_median}')



