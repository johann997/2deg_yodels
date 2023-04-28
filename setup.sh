#!/bin/bash

echo "STARTING SETUP"
printf %"$COLUMNS"s |tr " " "-"
printf %"$COLUMNS"s |tr " " "-"

# create environment
conda create  --name 2deg_yodels_env
printf %"$COLUMNS"s |tr " " "-"
echo "Environment Created"
printf %"$COLUMNS"s |tr " " "-"

conda activate 2deg_yodels_env
printf %"$COLUMNS"s |tr " " "-"
echo "Environment Activated"
printf %"$COLUMNS"s |tr " " "-"

conda install -c anaconda python
printf %"$COLUMNS"s |tr " " "-"
echo "Python Installed"
printf %"$COLUMNS"s |tr " " "-"

# Install kwant (using pip need to manually install MUMPS)
conda install kwant -c conda-forge
printf %"$COLUMNS"s |tr " " "-"
echo "kwant installed"
printf %"$COLUMNS"s |tr " " "-"

# Install required packages using pip :: pip freeze > requirements.txt
pip install -r requirements.txt
printf %"$COLUMNS"s |tr " " "-"
echo "Packages Installed"
printf %"$COLUMNS"s |tr " " "-"

printf %"$COLUMNS"s |tr " " "-"
printf %"$COLUMNS"s |tr " " "-"
echo "FINISHED SETUP"