import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import (
    QuantStub, 
    DeQuantStub, 
    get_default_qconfig, 
    get_default_qat_qconfig,
    prepare, 
    convert, 
    quantize_dynamic,
    QConfig,
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    default_per_channel_weight_observer,
)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import (
    prepare_fx, 
    convert_fx,
    prepare_qat_fx,
)

import torch.nn as nn
from transformers import (
    AutoModelForAudioClassification, 
    Wav2Vec2FeatureExtractor
)
from src.data_loader import get_data_loaders
import logging
from tqdm import tqdm
from src.mix_precision_quant import (
    run_calibration,
    FEWrapper,
    QuantizableFeatureExtractor,
    prepare_model_for_quant,
    prepare_model_for_quant_conv_only_fx,
    convert_frontend,
    convert_frontend_fx,
    evaluate,
    prepare_feature_extractor_for_qat_fx, 
    train_qat_feature_extractor,
    FeatureLoss
)
import torch.optim as optim
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from src.mix_precision_quant import FeatureLoss
import functools
import argparse

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX format.")
    parser.add_argument("--alpha", help="Temperature scaling", required=True, type=float)
    args = parser.parse_args()
    alpha = args.alpha

    data_config = {
    "raw_dir": "./data",
    "sample_rate": 16000,
    "batch_size": 256,
    "eval_batch_size": 4,
    "inference_batch_size": 4
    }
    train_loader, test_loader, validation_loader = get_data_loaders(data_config)

    model_baseline = AutoModelForAudioClassification.from_pretrained("./wav2vec2_finetuned_models/checkpoint-6630")
    model_04 = AutoModelForAudioClassification.from_pretrained("./wav2vec2_finetuned_models/checkpoint-6630")

    example_inputs = (torch.randn(1, 1, 16000),)  # adjust shape to your real input
    model_04 = prepare_feature_extractor_for_qat_fx(model_04, example_inputs, qat_qconfig=None)

    # freeze encoder layers
    for param in model_04.wav2vec2.encoder.parameters():
        param.requires_grad = False

    feature_extractor_fp32 = model_baseline.wav2vec2.feature_extractor.eval().cpu()
    feature_extractor_qat = model_04.wav2vec2.feature_extractor.train().cpu()

    custom_qat_qconfig = QConfig(
        activation=functools.partial(
            FusedMovingAvgObsFakeQuantize,
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,   # 8-bit full range
            reduce_range=False
        ),
        weight=functools.partial(
            FusedMovingAvgObsFakeQuantize,
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False
        )
    )
    qconfig_mapping = QConfigMapping().set_global(None).set_object_type(
        nn.Conv1d, custom_qat_qconfig
    )
    example_input = torch.randn(1, 1, 16000)  # 1 sec of audio at 16kHz
    feature_extractor_qat = prepare_qat_fx(
        feature_extractor_qat, 
        qconfig_mapping, 
        example_inputs=(example_input,)
    )

    feature_extractor_qat = train_qat_feature_extractor(
        teacher_model=feature_extractor_fp32,
        student_model=feature_extractor_qat,
        dataloader=validation_loader, 
        criterion=FeatureLoss(alpha=alpha),
        optimizer=optim.Adam(feature_extractor_qat.parameters(), lr=1e-4, weight_decay=1e-6),
        num_epochs=5)
        
    feature_extractor_q = convert_fx(feature_extractor_qat.eval())

    # 9. Plug back into wav2vec2
    model_04.wav2vec2.feature_extractor = feature_extractor_q

    model_04.eval()
    acc_04 = evaluate(model_04, test_loader)
    print(f"Quantized model (custom qconfig, exclude layer_norm) accuracy: {acc_04:.4f}")