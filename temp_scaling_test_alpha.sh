#!/bin/bash

#BSUB -J audioml_onnx_dynamic_quantize
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
#BSUB -N
#BSUB -o audioml_onnx_dynamic_quantize_%J.out
#BSUB -e audioml_onnx_dynamic_quantize_%J.err

module unload python3
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate audioml

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate audioml

python -m pip install -e . -q

python src/mix_precision_QAT.py --alpha 0.1
python src/mix_precision_QAT.py --alpha 0.3
python src/mix_precision_QAT.py --alpha 0.5
python src/mix_precision_QAT.py --alpha 0.7
python src/mix_precision_QAT.py --alpha 0.9