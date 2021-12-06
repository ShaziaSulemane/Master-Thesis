#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
 
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <Install Folder>"
    exit
fi
folder="$1"
user="smartcam"
passwd="smartcam"
 
echo "** Install requirement"
sudo apt-get update
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install python-numpy python3-numpy
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install libv4l-dev v4l-utils qv4l2 v4l2ucp
sudo apt-get install curl
sudo apt-get update
 
echo "** Download opencv-4.5.0"
cd $folder
curl -L https://github.com/opencv/opencv/archive/4.5.0.zip -o opencv-4.5.0.zip
curl -L https://github.com/opencv/opencv_contrib/archive/4.5.0.zip -o opencv_contrib-4.5.0.zip
unzip opencv-4.5.0.zip 
unzip opencv_contrib-4.5.0.zip 
cd opencv-4.5.0/
 
echo "** Building..."
mkdir release
cd release/
cmake -D WITH_CUDA=ON -D CUDA_ARCH_BIN=6.1 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python2=ON -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUBLAS=1 -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON ..

make -j8
sudo make install
sudo ldconfig
sudo apt-get install -y python-opencv python3-opencv
 
echo "** Install opencv-4.5.0 successfully"

