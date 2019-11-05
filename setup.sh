#!/bin/bash

conda create -n py36 python=3.6
source activate py36
pip install -r requirements.txt
