from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from project.data import DisasterTweetData
from project.model import DisasterTweetBertModel
from pytorch_lightning.callbacks import ModelCheckpoint


from transformers import DistilBertTokenizerFast
import torch
import random
import numpy as np

import time

import logging
import hydra



@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="train_config",
)
def train(config):

    hparams = config.Hyperparameters
    data_config = config.Data
    torch.manual_seed(hparams.seed)
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)

    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Load the dataset
    train_dataset = DisasterTweetData(data_path=data_config.train.filepath, tokenizer=tokenizer)
    val_dataset = DisasterTweetData(data_path=data_config.valid.filepath, tokenizer=tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=hparams.Batch_Size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams.Batch_Size)

    # Initialize the model
    model = DisasterTweetBertModel(lr=hparams.learning_rate)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename="disaster_tweet_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Initialize a trainer
    trainer = Trainer(max_epochs=hparams.Epoch,limit_train_batches=0.1, callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
