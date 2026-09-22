from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast

from project.model import DisasterTweetBertModel


app = FastAPI(
    title="Disaster Tweet Classifier",
    description="API for classifying tweets as disaster-related or not.",
    version="1.0.0",
)


# Model and tokenizer are loaded once when the API starts.
MODEL_PATH = Path("artifacts/disaster_tweet_model:v0/disaster_tweet_model.ckpt")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DisasterTweetBertModel.load_from_checkpoint(MODEL_PATH)
model.eval()


class TweetRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Disaster Tweet Classifier API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TweetRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()

    label = "disaster" if prediction == 1 else "not disaster"

    return PredictionResponse(
        prediction=prediction,
        label=label,
        confidence=confidence,
    )
