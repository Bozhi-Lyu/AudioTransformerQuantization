from transformers import AutoFeatureExtractor
import numpy as np
from src.data_loader import get_datasets
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import yaml

with open("configs/wav2vec2_Finetune.yaml", "r") as f:
    config = yaml.safe_load(f)
train_dataset, test_dataset, validate_dataset, all_labels = get_datasets(config["data"])

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=len(all_labels))

training_args = TrainingArguments(
    output_dir="wav2vec2_finetuned_models",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=config["train"]["learning_rate"],
    per_device_train_batch_size=config["train"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
    per_device_eval_batch_size=config["train"]["per_device_eval_batch_size"],
    num_train_epochs=config["train"]["num_train_epochs"],
    warmup_ratio=config["train"]["warmup_ratio"],
    logging_steps=config["train"]["logging_steps"],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
