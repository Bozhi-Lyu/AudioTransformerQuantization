from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import yaml
from src.data_loader import get_data_loaders
import logging

with open("configs/wav2vec2_Finetune.yaml", "r") as f:
    config = yaml.safe_load(f)
_, test_loader, _ = get_data_loaders(config["data"])

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2Processor

model = AutoModelForAudioClassification.from_pretrained("./wav2vec2_finetuned_models/checkpoint-6630")
feature_extractor = AutoFeatureExtractor.from_pretrained("./wav2vec2_finetuned_models/checkpoint-6630")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

correct = 0
total = 0

logger = logging.getLogger("evaluate_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

for batch in test_loader:
    input_values = batch[0].squeeze(1) # torch.Size([256, 16000])
    labels = batch[1].to(device) # torch.Size([256])

    inputs = processor(
        input_values.numpy(),
        sampling_rate=config["data"]["sample_rate"],
        return_tensors="pt",
        padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    correct += (predictions == labels).sum().item()

    total += labels.size(0)
    logger.info(f"Processed {total} samples, accuracy so far: {correct / total:.4f}")

accuracy = correct / total
logger.info(f"Test Accuracy: {accuracy:.4f}")