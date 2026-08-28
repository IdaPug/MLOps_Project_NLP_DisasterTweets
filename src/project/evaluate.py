from torch.utils.data import DataLoader
from project.data import DisasterTweetData
from project.model import DisasterTweetBertModel
from transformers import DistilBertTokenizerFast
import torch
import hydra


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="evaluate_config",
)

def evaluate(config):

    hparams = config.Hyperparameters
    paths = config.Paths

    # Load the model
    model = DisasterTweetBertModel.load_from_checkpoint(paths.model_path)
    model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Load the dataset
    dataset = DisasterTweetData(data_path=paths.data_path, tokenizer=tokenizer)

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=hparams.Batch_Size,shuffle=True)

    # Evaluate the model
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            if total_predictions >= hparams.Max_Eval:
                print(f"Reached maximum evaluation limit of {hparams.Max_Eval} samples.")
                break

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()
