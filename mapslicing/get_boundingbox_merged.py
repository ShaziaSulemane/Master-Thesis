#!/usr/bin/python3
import cv2
import argparse
import numpy as np
import os
import re

user = os.environ.get('USER')

parser = argparse.ArgumentParser(description='Generate a transparent orthophoto with labels.')
parser.add_argument("-f", "--file", type=str, default='/home/'+user+'/Documents/image-split-with-overlap/images/Split/mapCIR.txt', help='File path with image positions.')
parser.add_argument("-b", "--bbox", type=str, default='/home/'+user+'/Documents/image-split-with-overlap/images/Split/mapCIR.txt', help='File path with bounding boxes positions.')
parser.add_argument("-o", "--outdir", type=str, default='', help='Output file directory. Empty if same path as file path.')
parser.add_argument("-c", "--color", nargs=4, type=int, default= [204, 0, 204, 255], help="Color of bounding box [B G R alpha]")
parser.add_argument("-w", "--width", type=int, default= 5, help="Witdth of bounding box")
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

if not args.bbox:
    print("Please specify the path to the file with images positions.")
    exit()

bbox_path = args.bbox
color = args.color
width = args.width

if not os.path.exists(bbox_path):
	print("File "+bbox_path+" does not exist.")
	exit()

img_info = {}

with open(bbox_path, 'r') as result_file:
	for line in result_file:
		if(line.find('left_x') > -1):
			temp = re.findall(r'\d+', line) 
			res = list(map(int, temp))

			if not img_name in img_info:
				img_info[img_name] = []
			img_info[img_name].insert(0, (res[2], res[2]+res[4], res[1], res[1]+res[3]))
		if(line.find('Image Path') > -1):
			temp_line = line

			img_name = temp_line.split(':')[1][1:]
			if os.path.exists(img_name):
				img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
				out = img
				out[:,:,:]=[0,0,0,0]

				cut_name, extension = os.path.splitext(img_name)
				img_name = os.path.basename(img_name)

if not args.file:
    print("Please specify the path to the file with images positions.")
    exit()

file_path = args.file

if not os.path.exists(file_path):
	print("File "+file_path+" does not exist.")
	exit()

cut_name, extension = os.path.splitext(file_path)

if not args.outdir:
	outdir = os.path.dirname(cut_name)
else:
	outdir = args.outdir
	os.makedirs(outdir, 0o666, True)

with open(file_path, 'r') as result_file:
	for line in result_file:
		if(line[0] == '$'):
			if(line.find('Original') > -1):
				temp = re.findall(r'\d+', line) 
				out_sizes = list(map(int, temp))
				out = np.zeros((out_sizes[0],out_sizes[1],4), np.uint8)
			if(line.find('Sliced') > -1):
				temp = re.findall(r'\d+', line) 
				sizes = list(map(int, temp))
				#print(sizes)
			if(line.find('Overlap') > -1):
				overlap = line.split('\t')[1]
				overlap = int(overlap[:-1])
				#print(overlap)
		else:
			content = line.split('\t')
			i = int(content[1])
			j = int(content[2][:-1])
			img_name = content[0]
			if os.path.exists(img_name):
				img_name = os.path.basename(img_name)
				if img_name in img_info:
					img_info[img_name].insert(0, (i,j))

bbox = []

for key in img_info:
	offset = img_info[key][0]
	data = img_info[key][1:]
	for box in data:
		bbox.insert(0,(box[0]+offset[0], box[1]+offset[0], box[2]+offset[1], box[3]+offset[1]))

def rectOverlap(rect1, rect2):
	# The rectangles do not overlap at all
	if(rect2[0] > rect1[1] or rect2[1] < rect1[0] or rect2[2] > rect1[3] or rect2[3] < rect1[2] ):
		return False
	else:
		overlap_x = min(rect2[1],rect1[1]) - max(rect2[0],rect1[0]);
		overlap_y = min(rect2[3],rect1[3]) - max(rect2[2],rect1[2]);
		area_overlap = overlap_x * overlap_y;

		area_rect1 = (rect1[1]-rect1[0]) * (rect1[3]-rect1[2]);
		area_rect2 = (rect2[1]-rect2[0]) * (rect2[3]-rect2[2]);

		# The rectangles overlap by less than 75% of either's area
		if ((area_overlap < 0.25*area_rect1) and (area_overlap < 0.25*area_rect2)):
			return False

		# The rectangels overlap
		return True

idx = 0
idy = 1
while(idx < len(bbox)):
	box1 = bbox[idx]
	while(idy < len(bbox)):
		box2 = bbox[idy]
		if(rectOverlap(box1, box2)):
			bbox[idx] = (min(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), max(box1[3], box2[3]))
			bbox.remove(box2)
		else:
			idy = idy+1
	idx = idx+1
	idy = idx+1

def draw_bbox(img, xinitial, xfinal, yinitial, yfinal, width, color):
		if(xinitial < width):
			xinitial = width
		if(yinitial < width):
			yinitial = width
		if(img.shape[0]-xfinal < width):
			xfinal = img.shape[0]-width
		if(img.shape[1]-yfinal < width):
			yfinal = img.shape[1]-width
		img[ xinitial-width:xfinal+width, yinitial-width:yinitial] = color
		img[ xinitial-width:xfinal+width, yfinal:yfinal+width] = color
		img[ xinitial-width:xinitial, yinitial:yfinal] = color
		img[ xfinal:xfinal+width, yinitial:yfinal] = color

for box in bbox:
	draw_bbox(out, box[0], box[1], box[2], box[3], width, color)

cv2.imwrite(cut_name+'_merged.png', out)