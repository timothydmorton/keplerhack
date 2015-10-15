#!/bin/bash

python simpop.py $1
python inferpop.py $1/synthetic_kois_single.h5
python inferpop.py $1/synthetic_kois_binaries.h5

