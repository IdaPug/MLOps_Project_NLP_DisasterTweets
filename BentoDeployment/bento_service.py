from pathlib import Path

import numpy as np
import onnxruntime as ort
import bentoml
from transformers import DistilBertTokenizerFast


MODEL_PATH = Path("artifacts/disaster_tweet_model_onnx/model.onnx")


@bentoml.service
class DisasterTweetService:
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

    @bentoml.api
    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        outputs = self.session.run(
            ["logits"],
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        logits = outputs[0]

        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        prediction = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(probabilities[0, prediction])

        label = "disaster" if prediction == 1 else "not disaster"

        return {
            "prediction": prediction,
            "label": label,
            "confidence": confidence,
        }
