import pytorch_lightning as pl
import torch
from transformers import  DistilBertModel

# Define the model
class DisasterTweetBertModel(pl.LightningModule):
    """A simple BERT-based model for binary classification of disaster tweets."""

    def __init__(self, lr = 1e-5):
        super().__init__()
        self.save_hyperparameters()

        # Pretrained DistilBERT model
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Simple Classifier on top of BERT. Class 0: Not Disaster, Class 1: Disaster
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 2)

        # loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        # Bert
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get CLS token
        cls_output = outputs.last_hidden_state[:, 0, :]

        # pass through classifier
        logits = self.classifier(cls_output)
        return logits

    def training_step(self, batch, batch_idx):
        # inputs
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # forward pass
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # log
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # inputs
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # forward pass
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # accuracy calculation
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # log
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
