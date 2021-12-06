# !/usr/bin/python3
import argparse
import os
import random
import shutil

ORIGINAL_PATH = "/home/shazia/Documents/YOLO/maps_2"
TEST_PATH = "/home/shazia/Documents/YOLO/Mask-RCNN/test/"
TRAIN_PATH = "/home/shazia/Documents/YOLO/Mask-RCNN/train/"

# ADICIONAR VARIAVEIS PARA DETERMINAR PERCENTAGEM TRAIN-TEST
percentage = 10

for folder in os.listdir(ORIGINAL_PATH):
    for file in os.listdir(os.path.join(ORIGINAL_PATH, folder)):
        rdm = random.randint(0, 100)
        if rdm < percentage:
            dest = shutil.move(os.path.join(os.path.join(ORIGINAL_PATH, folder), file), TEST_PATH + folder + '/')
        else:
            dest = shutil.move(os.path.join(os.path.join(ORIGINAL_PATH, folder), file), TRAIN_PATH + folder + '/')