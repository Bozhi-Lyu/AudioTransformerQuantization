#!/bin/bash
#BSUB -J audioml_finetune
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 18:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
#BSUB -o audioml_finetune_%J.out
#BSUB -e audioml_finetune_%J.err

module unload python3
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate audioml

python3 -m pip install -e . -q
echo "Starting evaluate checkpoints..."
python3 src/evaluate.py