#!/bin/bash

#BSUB -J temperature_alpha_test[1-5]
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o alpha%J.out
#BSUB -e alpha%J.err

module unload python3
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate audioml

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate audioml

python -m pip install -e . -q

case $LSB_JOBINDEX in
    1) ALPHA=0.1 ;;
    2) ALPHA=0.3 ;;
    3) ALPHA=0.5 ;;
    4) ALPHA=0.7 ;;
    5) ALPHA=0.9 ;;
esac

echo "Running with alpha=$ALPHA"
python src/mix_precision_QAT.py --alpha $ALPHA