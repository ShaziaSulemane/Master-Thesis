# split images
# find angles
# delete images
# rotate original image
# re split images

# !/usr/bin/python3
import argparse
import os
import random

TXT_PATH = "/home/ssulemane/Documents/YOLO/vine-gap/data/"
PATH = "/home/ssulemane/Documents/YOLO/vine-gap/data/obj/train/"
FILE_NAME_TRAIN = "train"
FILE_NAME_TEST = "test"
FORMAT = ".txt"

# ADICIONAR VARIAVEIS PARA DETERMINAR PERCENTAGEM TRAIN-TEST
percentage = 10


# TRAIN COM TODOS OS TIPOS DE FICHEIROS
for folder in os.listdir(PATH):
    train_file = open(TXT_PATH + '/' + FILE_NAME_TRAIN + '_' + folder + '.txt', 'w+')
    test_file = open(TXT_PATH + '/' + FILE_NAME_TEST + '_' + folder + '.txt', 'w+')
    for img_name in os.listdir(os.path.join(PATH, folder)):
        number = random.randint(0, 100)
        split_img_name = img_name.split(".")
        if split_img_name[1] != 'txt':
            if number >= percentage:
                train_file.write(os.path.join(PATH, folder, img_name) + '\n')
            else:
                test_file.write(os.path.join(PATH, folder, img_name) + '\n')
