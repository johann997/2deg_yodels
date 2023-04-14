#!/bin/bash


# create environment
conda create  --name 2deg_yodel
conda activate 2deg_yodel
conda install -c anaconda python



# Install required packages using pip
pip install -r requirements.txt

