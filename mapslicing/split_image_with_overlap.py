#!/usr/bin/python3
import cv2
import argparse
import numpy as np
import os

user = os.environ.get('USER')

parser = argparse.ArgumentParser(description='Slicing a complete orthophoto into smaller images.')
parser.add_argument("-f", "--file", type=str,
                    default='/home/' + user + '/Documents/image-split-with-overlap/images/image.png', help='File path')
parser.add_argument("-W", "--width", type=int, default='608', help='Width of cropped image')
parser.add_argument("-H", "--height", type=int, default='608', help='Height of cropped image')
parser.add_argument("-O", "--overlap", type=int, default='25', help='Overlap percentage (int value)')
parser.add_argument("-o", "--outdir", type=str, default='',
                    help='Output file directory. Empty if same path as original photo.')
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

if not args.file:
    print("Please specify the path to the images.")
    exit()

path_to_img = args.file

if not os.path.exists(path_to_img):
    print("File " + path_to_img + " does not exist.")
    exit()

cut_name, extension = os.path.splitext(path_to_img)

if not extension.lower() in ['.jpg', '.png', '.jpeg', '.tiff', '.tif']:
    print("Please check image extension.")
    exit()

original_name = os.path.basename(cut_name)

if not args.outdir:
    outdir = os.path.dirname(cut_name)
else:
    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir, 0o777)

img = cv2.imread(path_to_img, cv2.IMREAD_UNCHANGED)
img_h, img_w, _ = img.shape
split_width = args.width
split_height = args.height

if split_width > img_w or split_height > img_h:
    print("The original image is smaller than output size.")
    exit()

overlap = args.overlap / 100.0


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


X_points = start_points(img_w, split_width, overlap)
Y_points = start_points(img_h, split_height, overlap)

count = 0
name = outdir + '/' + original_name + '_splitted'
frmt = extension  # 'jpeg'

text_file = open(outdir + '/' + original_name + '.txt', "w+")
text_file.write("$ Original Image:\t[%i %i]px\n" % (img_h, img_w))
text_file.write("$ Sliced Images:\t[%i %i]px\n" % (split_height, split_width))
text_file.write("$ Overlap:\t%i\n" % (args.overlap))
text_file.write("$ File Name \t Column \t Rowrgb4_splitted_188.png\n")

for idy, i in enumerate(Y_points):
    for idx, j in enumerate(X_points):
        split = img[i:i + split_height, j:j + split_width]
        if np.sum(split[:, :, 3] > 0) / float(split_width * split_height) < overlap:
            # split[:,:,0]=255
            continue
        img_name = '{}_{}{}'.format(name, count, frmt)
        cv2.imwrite(img_name, split)
        text_file.write("%s\t%i\t%i\n" % (img_name, i, j))
        count += 1

