#!/usr/bin/python3
import cv2
import argparse
import numpy as np
import os
import re

user = os.environ.get('USER')

parser = argparse.ArgumentParser(description='Merge smaller images into a complete orthophoto.')
parser.add_argument("-f", "--file", type=str, default='/home/'+user+'/Documents/image-split-with-overlap/images/Split/mapCIR.txt', help='File path with image positions.')
parser.add_argument("-o", "--outdir", type=str, default='', help='Output file directory. Empty if same path as file path.')
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

if not args.file:
    print("Please specify the path to the file with images positions.")
    exit()

file_path = args.file

if not os.path.exists(file_path):
	print("File "+file_path+" does not exist.")
	exit()

cut_name, extension = os.path.splitext(file_path)

original_name = os.path.basename(cut_name)
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
			img_name, extension = os.path.splitext(content[0])
			if os.path.exists(img_name+'_filtered'+extension):
				img = cv2.imread(img_name+'_filtered'+extension, cv2.IMREAD_UNCHANGED)
				out[i:i+sizes[0], j:j+sizes[1]] = cv2.bitwise_or(out[i:i+sizes[0], j:j+sizes[1]], img)
			#print(content)
cv2.imwrite(cut_name+'_merged'+extension, out)