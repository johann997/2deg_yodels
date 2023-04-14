#!/bin/bash

# create environment
conda create  --name 2deg_yodels_env
conda activate 2deg_yodels_env
conda install -c anaconda python

# Install required packages using pip
pip install -r requirements.txt

