import torch
from transformers import AutoModelForAudioClassification
from transformers import AutoFeatureExtractor
import numpy as np

model = AutoModelForAudioClassification.from_pretrained("./wav2vec2_finetuned_models/checkpoint-6630")
model.eval()

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

dummy_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
inputs = feature_extractor(dummy_input.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

torch.onnx.export(
    model,
    (inputs["input_values"]),
    "models/wav2vec2_finetuned.onnx",
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={"input_values": {0: "batch_size", 1: "sequence_length"},
                  "logits": {0: "batch_size"}},
    opset_version=14
)
