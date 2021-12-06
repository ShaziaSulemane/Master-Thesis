#!/usr/bin/python3
import re
import cv2
import os
import argparse

user = os.environ.get('USER')

parser = argparse.ArgumentParser(description='Generate images with bounding boxes from YOLO output')
parser.add_argument("-f", "--file", type=str, default='/home/'+user+'/Documents/image-split-with-overlap/images/Split/mapCIR.txt', help='File path with bounding boxes positions.')
#parser.add_argument("-o", "--outdir", type=str, default='', help='Output file directory. Empty if same path as file path.')
parser.add_argument("-c", "--color", nargs=4, type=int, default= [204, 0, 204, 255], help="Color of bounding box [B G R alpha]")
parser.add_argument("-w", "--width", type=int, default= 5, help="Witdth of bounding box")
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

if not args.file:
    print("Please specify the path to the file with images positions.")
    exit()

file_path = args.file
color = args.color
width = args.width

if not os.path.exists(file_path):
	print("File "+file_path+" does not exist.")
	exit()

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


with open(file_path, 'r') as result_file:
	for line in result_file:
		if(line.find('left_x') > -1):
			temp = re.findall(r'\d+', line) 
			res = list(map(int, temp))
			
			#print(img_name)
			#print(out_name)
			#print(res)
			draw_bbox( out, res[2], res[2]+res[4], res[1], res[1]+res[3], width, color) #out[res[2]:res[2]+res[4], res[1]:res[1]+res[3], 3]=[255
			cv2.imwrite(out_name, out)
		if(line.find('Image Path') > -1):
			temp_line = line

			img_name = temp_line.split(':')[1][1:]
			if os.path.exists(img_name):
				img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
				out = img
				out[:,:,:]=[0,0,0,0]

				cut_name, extension = os.path.splitext(img_name)
				out_name = cut_name+'_filtered'+extension