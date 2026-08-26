from pathlib import Path
import pandas as pd
import torch

import typer
from torch.utils.data import Dataset


class DisasterTweetData(Dataset):
    def __init__(self, data_path: Path, tokenizer=None,maxlen=128):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.data = pd.read_csv(data_path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, "text"])
        label = self.data.loc[idx, "target"]

        if self.tokenizer:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.maxlen,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
            )

            return {
                "text": text,
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long)
            }



def preprocess(data_path: Path, output_path: Path, train_frac: float = 0.8):
    # load the data
    data = pd.read_csv(data_path)

    # split the data into train and validation sets
    train = data.sample(frac=train_frac, random_state=42)
    valid = data.drop(train.index)

    # make output directory and save
    output_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_path / "train.csv", index=False)
    valid.to_csv(output_path / "valid.csv", index=False)


if __name__ == "__main__":
    # run the preprocessing function
    preprocess(Path("data/train.csv"), Path("data/processed_data"), train_frac=0.8)
