from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from project.data import DisasterTweetData
from project.model import DisasterTweetBertModel

from transformers import DistilBertTokenizerFast
import torch

import time



def train(Epochs=1, Batch_Size=16, lr=1e-5):
    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Load the dataset
    train_dataset = DisasterTweetData(data_path="data/processed_data/train.csv", tokenizer=tokenizer)
    val_dataset = DisasterTweetData(data_path="data/processed_data/valid.csv", tokenizer=tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_Size)

    # Initialize the model
    model = DisasterTweetBertModel(lr=lr)

    # Initialize a trainer
    trainer = Trainer(max_epochs=Epochs,limit_train_batches=0.1)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # get time stamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save the model
    torch.save(model.state_dict(), f"models/disaster_tweet_model_{timestamp}.pth")

if __name__ == "__main__":
    train()
