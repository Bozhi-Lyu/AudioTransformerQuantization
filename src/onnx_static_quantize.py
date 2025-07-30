from src.data_loader import get_data_loaders
import torch
from transformers import AutoFeatureExtractor
import numpy as np
import yaml
import onnx
import argparse

from onnxruntime.quantization import QuantFormat, quantize_static, CalibrationDataReader

class DataReader(CalibrationDataReader):
    def __init__(self, dataloader, max_batches=100):
        self.dataloader = iter(dataloader)
        self.iterator = None
        self.max_batches = max_batches

    def get_next(self):
        if self.iterator is None:
            self.iterator = self._yield_batches()
        return next(self.iterator, None)

    def _yield_batches(self):
        for i, batch in enumerate(self.dataloader):
            if i >= self.max_batches:
                break
            input_tensor = batch[0].squeeze(1).cpu().numpy()
            yield {"input_values": input_tensor}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Static quantization with ONNX Runtime.")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--config", required=False, default="configs/wav2vec2_Finetune.yaml", type=str)
    parser.add_argument("--per_channel", required=False, type=bool, default=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    _, _, validate_loader = get_data_loaders(config["data"])

    onnx_static = onnx.load("./models/wav2vec2_finetuned.onnx")
    matmul_nodes = [node for node in onnx_static.graph.node if node.op_type == "MatMul"]
    conv_nodes = [node for node in onnx_static.graph.node if node.op_type == "Conv"]
    Gemm_nodes = [node for node in onnx_static.graph.node if node.op_type == "Gemm"]
    Add_nodes = [node for node in onnx_static.graph.node if node.op_type == "Add"]
    Mul_nodes = [node for node in onnx_static.graph.node if node.op_type == "Mul"]
    Div_nodes = [node for node in onnx_static.graph.node if node.op_type == "Div"]
    ReduceMean_nodes = [node for node in onnx_static.graph.node if node.op_type == "ReduceMean"]
    nodes_to_quantize = matmul_nodes + conv_nodes + Gemm_nodes + Add_nodes + Mul_nodes + Div_nodes + ReduceMean_nodes

    quantize_static(
        model_input=args.input,
        model_output=args.output,
        nodes_to_quantize=nodes_to_quantize,
        calibration_data_reader=DataReader(validate_loader),
        quant_format=QuantFormat.QDQ, # Models quantized by quantize_static are in QDQ format
        per_channel=args.per_channel,
    )

    print("ONNX Runtime Static quantization successful.")