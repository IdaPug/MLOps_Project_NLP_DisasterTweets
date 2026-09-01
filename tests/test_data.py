from torch.utils.data import Dataset
from project.data import DisasterTweetData
import pandas as pd
import torch
import os
from transformers import DistilBertTokenizerFast


from tests import _PATH_DATA

def get_tokenizer():
    # Load the tokenizer
    tokenizer  = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    return tokenizer

def make_dummy_csv():
    # Create a dummy CSV file for testing
    import pandas as pd
    data = {
        'id': [0, 1],
        'keyword': ['disaster', 'not_disaster'],
        'location': ['location1', 'location2'],
        'text': ['This is a disaster tweet', 'This is not a disaster tweet'],
        'target': [1, 0]
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(_PATH_DATA, "dummy_data.csv")
    df.to_csv(file_path, index=False)
    return file_path

def test_dataset_loading():
    # Create a dummy CSV file
    dummy_csv_path = make_dummy_csv()
    tokenizer = get_tokenizer()

    # Load the dataset
    dataset = DisasterTweetData(data_path=dummy_csv_path, tokenizer=tokenizer)

    # Check if the dataset has the correct length
    assert len(dataset) == 2, "Dataset length should be 2 for the dummy data"

def test_dataset_item_structure():
    # Create a dummy CSV file
    dummy_csv_path = make_dummy_csv()
    tokenizer = get_tokenizer()

    # Load the dataset
    dataset = DisasterTweetData(data_path=dummy_csv_path, tokenizer=tokenizer)

    # Get an item from the dataset
    item = dataset[0]

    # Check if the item has the expected keys
    assert 'text' in item, "Dataset item should contain 'text' key"
    assert 'input_ids' in item, "Dataset item should contain 'input_ids' key"
    assert 'attention_mask' in item, "Dataset item should contain 'attention_mask' key"
    assert 'labels' in item, "Dataset item should contain 'labels' key"

    # Check the types of the values
    assert isinstance(item['text'], str), "Dataset item 'text' should be a string"
    assert isinstance(item['input_ids'], torch.Tensor), "Dataset item 'input_ids' should be a torch.Tensor"
    assert isinstance(item['attention_mask'], torch.Tensor), "Dataset item 'attention_mask' should be a torch.Tensor"
    assert isinstance(item['labels'], torch.Tensor), "Dataset item 'labels' should be a torch.Tensor"


    # Check the shapes of the tensors
    assert item['input_ids'].shape[0] == 128, "Dataset item 'input_ids' should have shape (128,)"
    assert item['attention_mask'].shape[0] == 128, "Dataset item 'attention_mask' should have shape (128,)"
    assert item["labels"].shape == torch.Size([]), "Dataset item 'labels' should have shape (1,)"


def test_tokenization():
    # Create a dummy CSV file
    dummy_csv_path = make_dummy_csv()

    tokenizer = get_tokenizer()
    # Load the dataset
    dataset = DisasterTweetData(data_path=dummy_csv_path, tokenizer=tokenizer)

    # Get an item from the dataset
    item = dataset[0]

    # Check the CLS token (first token) in the input_ids
    cls_token_id = tokenizer.cls_token_id
    assert item['input_ids'][0].item() == cls_token_id, "The first token in 'input_ids' should be the CLS token"
