import torch
import torch.nn as nn
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
from torch.ao.quantization.quantize_fx import (
    prepare_fx, 
    convert_fx,
    prepare_qat_fx,
)
import torch.ao.quantization as tq
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from transformers import (
    Wav2Vec2ForCTC, 
    AutoModelForAudioClassification, 
    AutoFeatureExtractor, 
    Wav2Vec2Processor, 
    Wav2Vec2FeatureExtractor
)
import functools
import argparse
from src.data_loader import get_data_loaders
import yaml
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Calibration for static quantization
def run_calibration(model: nn.Module, 
                    calibration_loader, 
                    feature_extractor,
                    max_batches: int = None) -> nn.Module:
    """Run a forward pass on calibration data to populate observers.
    Keep everything on CPU during calibration.
    """
    model.eval()
    processed_batches = 0
    with torch.no_grad():
        for batch in tqdm(calibration_loader, desc="Calibrating", unit="batch"):
            # Expect batch[0] shape: [B, 1, T]; move to CPU
            input_values = batch[0].squeeze(1).cpu().numpy()
            inputs = feature_extractor(
                input_values,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            # model expects input_values and optional attention_mask
            _ = model(inputs["input_values"], attention_mask=inputs.get("attention_mask", None))
            processed_batches += 1
            if max_batches is not None and max_batches > 0 and processed_batches >= max_batches:
                break

    logger.info("Calibration run finished (processed %d batches).", processed_batches)
    return model


class FEWrapper(nn.Module):
    """Wrap the conv frontend with Quant/DeQuant stubs.
    Input: float tensor -> QuantStub -> feature_extractor (quantized) -> DeQuantStub -> float tensor
    """
    def __init__(self, fe: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.fe = fe
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fe(x)
        x = self.dequant(x)
        return x


class QuantizableFeatureExtractor(nn.Module):
    def __init__(self, fe):
        super().__init__()
        self.quant = tq.QuantStub()
        self.conv_layers = fe.conv_layers  # keep original conv stack
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        for layer in self.conv_layers:  # iterate; ModuleList has no forward()
            x = layer(x)
        x = self.dequant(x)
        return x
 

def prepare_model_for_quant(model: nn.Module, qconfig=None):
    """
    Replace feature_extractor with wrapper and call prepare(...) on wrapper.
    Allows injecting a custom qconfig.
    """
    # Ensure CPU and evaluation mode
    torch.backends.quantized.engine = "fbgemm"  # x86
    model = model.to("cpu").eval()

    # Wrap the frontend conv stack
    wrapped_fe = FEWrapper(model.wav2vec2.feature_extractor)
    
    # Assign appropriate qconfig. fbgemm is typical for x86 CPU.
    wrapped_fe.qconfig = qconfig if qconfig else tq.get_default_qconfig("fbgemm")
    
    model.wav2vec2.feature_extractor = wrapped_fe

    # Prepare the wrapped frontend for static PTQ (insertion of observers)
    tq.prepare(model.wav2vec2.feature_extractor, inplace=True)
    logger.info("Prepared feature_extractor wrapper for static PTQ (observers inserted).")
    return model

def prepare_model_for_quant_conv_only_fx(model: nn.Module, qconfig: QConfig):
    """
    FX-based prepare for quantizing only Conv1d inside feature_extractor.
    - trace and prepare only the feature_extractor submodule (safer for non-traceable top-level model).
    - qconfig_mapping: global=None (skip everything) + nn.Conv1d -> qconfig (quantize convs).
    """
    torch.backends.quantized.engine = "fbgemm"
    model = model.to("cpu").eval()

    # symbolically trace & prepare the feature_extractor submodule only
    orig_fe = model.wav2vec2.feature_extractor
    example_inputs = (torch.randn(1, 1, 16000),)  

    qconfig = qconfig if qconfig else tq.get_default_qconfig("fbgemm")
    print("Using qconfig:", qconfig)
    qconfig_mapping = QConfigMapping().set_global(None).set_object_type(nn.Conv1d, qconfig)
    prepared_fe = prepare_fx(orig_fe, qconfig_mapping, example_inputs)

    # attach prepared fe back to the model so existing calibration helper can run
    model.wav2vec2.feature_extractor = prepared_fe
    logger.info("Prepared feature_extractor (FX) for conv-only PTQ (observers inserted).")
    return model

def prepare_feature_extractor_for_qat_fx(model: nn.Module, example_inputs: tuple, qat_qconfig=None):
    """
    Trace and prepare the model.wav2vec2.feature_extractor for QAT (FX mode).
    - example_inputs: tuple of example tensors used for tracing, e.g. (torch.randn(1,1,16000),)
    - qat_qconfig: QAT qconfig (if None, use default for fbgemm).
    Returns: model with model.wav2vec2.feature_extractor replaced by the prepared GraphModule.
    """
    torch.backends.quantized.engine = "fbgemm"
    model = model.to("cpu")
    model.eval()  # prepare_qat_fx will switch modules internally as needed

    if qat_qconfig is None:
        qat_qconfig = get_default_qat_qconfig("fbgemm")

    # Build QConfigMapping that quantizes only Conv1d and leaves everything else in fp32
    qconfig_mapping = QConfigMapping().set_global(None).set_object_type(nn.Conv1d, qat_qconfig)

    # Symbolically trace and prepare the feature_extractor for QAT
    orig_fe = model.wav2vec2.feature_extractor
    prepared_fe = prepare_qat_fx(orig_fe, qconfig_mapping, example_inputs)

    # Attach prepared feature extractor back to the model for QAT fine-tuning
    model.wav2vec2.feature_extractor = prepared_fe
    logger.info("Attached FX-prepared QAT feature_extractor to model.")
    return model

def convert_frontend(model: nn.Module):
    """Convert the prepared frontend wrapper to quantized ops."""
    tq.convert(model.wav2vec2.feature_extractor, inplace=True)
    logger.info("Converted feature_extractor to quantized implementation.")
    return model

def convert_frontend_fx(model: nn.Module):
    # convert the prepared feature_extractor (GraphModule) to quantized version
    prepared_fe = model.wav2vec2.feature_extractor
    quantized_fe = convert_fx(prepared_fe)
    model.wav2vec2.feature_extractor = quantized_fe
    logger.info("Converted feature_extractor (FX) -> quantized.")
    return model

def train_qat_feature_extractor(teacher_model: nn.Module,
                                student_model: nn.Module,
                                dataloader, 
                                criterion,
                                optimizer,
                                num_epochs: int = 10):
    teacher_model.eval()
    student_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        # tqdm progress bar for each epoch
        for batch in pbar:
            input_values = batch[0].squeeze(1).cpu()  # keep as tensor
            with torch.no_grad():
                teacher_out = teacher_model(input_values)

            student_out = student_model(input_values)

            # teacher_out = teacher_out / (teacher_out.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
            # student_out = student_out / (student_out.abs().max(dim=-1, keepdim=True)[0] + 1e-8)

            loss = criterion(student_out, teacher_out)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(avg_loss=f"{epoch_loss/(pbar.n+1):.6f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.6f}")

    return student_model


def evaluate(model: nn.Module, test_loader):
    """Evaluate model on test_loader. Model must be on CPU."""
    model = model.to("cpu").eval()
    correct, total = 0, 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
        for batch in pbar:
            input_values = batch[0].squeeze(1).cpu()
            labels = batch[1].cpu()

            outputs = model(input_values)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # live update accuracy in tqdm bar
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(acc=f"{acc:.4f}")

    final_acc = correct / total if total > 0 else 0.0
    logger.info("Final Test Accuracy: %.4f (samples=%d)", final_acc, total)
    return final_acc


class FeatureLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, student_out, teacher_out):
        # Flatten over time/freq dimensions if needed
        s = student_out.view(student_out.size(0), -1)
        t = teacher_out.view(teacher_out.size(0), -1)

        mse_loss = self.mse(s, t)
        # Remove normalization before MSE, Keep normalization for cosine similarity, 
        # but use raw outputs for MSE. This way, MSE captures magnitude information 
        # while cosine captures directional alignment.

        # Normalize to stabilize cosine similarity
        s = F.normalize(s, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)
        cos_loss = 1.0 - self.cos(s, t).mean()

        return self.alpha * mse_loss + (1 - self.alpha) * cos_loss
