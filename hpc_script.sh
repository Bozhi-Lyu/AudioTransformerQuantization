#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate audioml

python -m pip install -e . -q
python src/finetune.py