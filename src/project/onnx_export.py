from pathlib import Path

import torch
from transformers import DistilBertTokenizerFast

from project.model import DisasterTweetBertModel


CHECKPOINT = Path("artifacts/disaster_tweet_model:v0/disaster_tweet_model.ckpt")
OUTPUT_DIR = Path("artifacts/disaster_tweet_model_onnx")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = OUTPUT_DIR / "model.onnx"


# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load trained Lightning checkpoint
model = DisasterTweetBertModel.load_from_checkpoint(CHECKPOINT)
model.eval()


# Example inputs used only to define the ONNX graph
example = tokenizer(
    "There is a huge earthquake and people need help",
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=128,
)

input_ids = example["input_ids"]
attention_mask = example["attention_mask"]


# Export model
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
)

print(f"ONNX model saved to: {ONNX_PATH}")
