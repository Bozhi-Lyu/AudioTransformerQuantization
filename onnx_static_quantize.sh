#!/bin/bash

#BSUB -J audioml_onnx_static_quantize
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
#BSUB -o audioml_onnx_static_quantize_%J.out
#BSUB -e audioml_onnx_static_quantize_%J.err

module unload python3
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate audioml

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate audioml

python -m pip install -e . -q

# 1. Export the FP model to ONNX format
python src/onnx_export.py

# 2. Preprocess the ONNX model
python -m onnxruntime.quantization.preprocess \
    --input models/wav2vec2_finetuned.onnx \
    --output models/wav2vec2_finetuned_infer.onnx

# 3. Perform static quantization
python src/onnx_static_quantize.py \
    --input models/wav2vec2_finetuned_infer.onnx \
    --output models/wav2vec2_finetuned_int8.onnx \
    --per_channel True