# split images
# find angles
# delete images
# rotate original image
# re split images

# !/usr/bin/python3

import math

import cv2 as cv
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from scipy import ndimage

PATH = '/home/shazia/Documents/YOLO/maps_2'
OUTPATH = '/home/shazia/Documents/YOLO/'
angles = []
medians = []


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


for folder in os.listdir(PATH):
    img_outdir = os.path.join(OUTPATH, folder)

    print("**************************************************************")
    print(f"\nFirst Step: Splitting Images in folder {folder}")

    for imgdir in os.listdir(os.path.join(PATH, folder)):
        img_path = os.path.join(os.path.join(PATH, folder), imgdir)
        command = "python /home/shazia/Documents/YOLO/mapslicing/split_image_with_overlap.py -f " + img_path + " -o " + img_outdir
        os.system(command)

        # img_name = imgdir.split(".")
        # print("Second Step: Calculating Rotation Angles for " + imgdir)

        # for splitdir in os.listdir(os.path.join(OUTPATH, folder)):
        #     img_path = os.path.join(os.path.join(OUTPATH, folder), splitdir)
        #     img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        #     splitdir_name = splitdir.split("_")
        #     if splitdir.split(".")[1] != "png":
        #         continue
        #
        #     # print(splitdir + " " + str(img.shape))
        #     img_h, img_w, _ = img.shape
        #
        #     if img is not None and splitdir_name[0] == img_name[0] and np.sum(img[:, :, 3] > 0) / float(
        #             img_w * img_h) > 0.90:
        #         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #         ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
        #         lines = cv.HoughLinesP(thresh, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
        #
        #         # cv.imshow("thresh", thresh)
        #         # cv.waitKey(0)
        #
        #         if lines is not None:
        #
        #             for [[x1, y1, x2, y2]] in lines:
        #                 angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        #                 angles.append(angle)
        #                 cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
        #
        #             # if np.median(angles) != -90.0 and np.median(angles) != -45.0:
        #             medians.append(np.median(angles))
        #
        #             # print(medians)
        #
        #             # cv.imshow("lines", img)
        #             # cv.waitKey(0)
        #
        #             if not os.path.isdir(os.path.join(OUTPATH, folder)):
        #                 os.makedirs(os.path.join(OUTPATH, folder), 0o777)
        #
        # print("Third Step: Rotating all split Images\n")
        # final_median = np.median(medians)
        # medians.clear()
        #
        # for split in os.listdir(os.path.join(OUTPATH, folder)):
        #     img_path = os.path.join(os.path.join(OUTPATH, folder), split)
        #     img = cv.imread(img_path)
        #     split_name = split.split("_")
        #
        #     if img is not None and split_name[0] == img_name[0]:
        #         print("Rotating: " + split + f" by {final_median} degrees")
        #         rotated = rotate_image(img, final_median)
        #         cv.imwrite(img_path, rotated)

    print("**************************************************************")
