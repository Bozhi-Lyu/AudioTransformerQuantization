#!/bin/bash

#BSUB -J audioml_onnx_static_quantize
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 16:00
#BSUB -R "rusage[mem=16GB]"
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
    --output models/wav2vec2_finetuned_static_int8.onnx \
    --per_channel True

# "WARNING:root:Axis 1 is out-of-range for weight '/wav2vec2/feature_extractor/conv_layers.0/layer_norm/Constant_1_output_0' with rank 1"
# This warning means a 1D tensor can't be quantized per-channel along axis 1.
# It's automatically handled by falling back to per-tensor quantization.

# 4. Run inference with the quantized model
# python src/onnxRT_inference.py \
#     --model models/wav2vec2_finetuned.onnx \
#     --config configs/wav2vec2_Finetune.yaml
# python src/onnxRT_inference.py \
#     --model models/wav2vec2_finetuned_infer.onnx \
#     --config configs/wav2vec2_Finetune.yaml
python src/onnxRT_inference.py \
    --model models/wav2vec2_finetuned_static_int8.onnx \
    --config configs/wav2vec2_Finetune.yaml