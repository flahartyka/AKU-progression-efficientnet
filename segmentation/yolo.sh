#!/bin/bash

#delete stuff from .cache/torch/kernels
source myconda
mamba activate base
export YOLO_CONFIG_DIR=/data/flahartyka/

cd /data/flahartyka/yolo_cervical

python trainyolo.py



