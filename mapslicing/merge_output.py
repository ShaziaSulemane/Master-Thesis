#!/usr/bin/python3
import cv2
import argparse
import numpy as np
import os
import re

user = os.environ.get('USER')

parser = argparse.ArgumentParser(description='Merge smaller images into a complete orthophoto.')
parser.add_argument("-f", "--file", type=str, default='/home/'+user+'/Documents/image-split-with-overlap/images/Split/mapCIR_merged.png', help='File path with labels.')
parser.add_argument("-b", "--base", type=str, default='/home/'+user+'/Documents/image-split-with-overlap/images/map', help='Source image  path and name prefix.')
parser.add_argument("-o", "--outdir", type=str, default='', help='Output file directory. Empty if same path as file path.')
parser.add_argument(      "--ndvi", action="store_true", help='NDVI.')
parser.add_argument(      "--rgb",  action="store_true", help='RGB.')
parser.add_argument(      "--ndre", action="store_true", help='NDRE.')
parser.add_argument(      "--cir",  action="store_true", help='CIR.')
parser.add_argument(      "--ndwi", action="store_true", help='NDWI.')
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

label_path = args.file

if not os.path.exists(label_path):
	print("File "+label_path+" does not exist.")
	exit()

base_name = args.base

def merge_images(base, labels):
	if not os.path.exists(base):
		print("File "+base+" does not exist.")
		return False
	global img
	img = cv2.imread(base, cv2.IMREAD_UNCHANGED)
	lbl = cv2.imread(labels, cv2.IMREAD_UNCHANGED)
	mask = np.nonzero(lbl[:,:,3])
	img[mask] = lbl[mask]
	return True

def generate_image(img_path):
	# Generate Output image
	cut_name, extension = os.path.splitext(img_path)
	original_name = os.path.basename(cut_name)
	if not args.outdir:
		outdir = os.path.dirname(cut_name)
	else:
		outdir = args.outdir
		os.makedirs(outdir, 0o666, True)

	cv2.imwrite(outdir+'/'+original_name+'_withLabels'+extension, img)

if args.ndvi:
	out_file = base_name+'NDVI.png'
	if(merge_images(out_file, label_path)):
		generate_image(out_file)

if args.rgb:
	out_file = base_name+'RGB.png'
	if(merge_images(out_file, label_path)):
		generate_image(out_file)
	out_file = base_name+'RGBGamma.png'
	if(merge_images(out_file, label_path)):
		generate_image(out_file)

if args.ndre:
	out_file = base_name+'NDRE.png'
	if(merge_images(out_file, label_path)):
		generate_image(out_file)

if args.cir:
	out_file = base_name+'CIR.png'
	if(merge_images(out_file, label_path)):
		generate_image(out_file)

if args.ndwi:
	out_file = base_name+'NDWI.png'
	if(merge_images(out_file, label_path)):
		generate_image(out_file)