import torch
import pytorch_lightning as pl

from project.model import DisasterTweetBertModel


def test_model_construction():

    model = DisasterTweetBertModel()
    assert isinstance(model, pl.LightningModule), "Model should be an instance of pl.LightningModule"

    # check components
    assert hasattr(model, 'bert'), "Model should have a 'bert' attribute"
    assert hasattr(model, 'classifier'), "Model should have a 'classifier' attribute"
    assert hasattr(model, 'loss_fn'), "Model should have a 'loss_fn' attribute"

    # check dimensions of classifier
    assert model.classifier.in_features == model.bert.config.hidden_size, "Classifier input features should match BERT hidden size"
    assert model.classifier.out_features == 2, "Classifier output features should be 2 for binary classification"


def test_model_forward_pass():
    model = DisasterTweetBertModel()
    model.eval()  # Set the model to evaluation mode

    # Create dummy input data
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, model.bert.config.vocab_size, (batch_size, seq_length))  # Random token IDs
    attention_mask = torch.ones((batch_size, seq_length))  # All tokens are attended to

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    # Check output shape
    assert logits.shape == (batch_size, 2), "Output logits should have shape (batch_size, 2)"

    # check output values are finite
    assert torch.isfinite(logits).all(), "Output logits should be finite values"


def test_training_step():
    model = DisasterTweetBertModel()
    model.train()  # Set the model to training mode

    # Create dummy input data
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, model.bert.config.vocab_size, (batch_size, seq_length))  # Random token IDs
    attention_mask = torch.ones((batch_size, seq_length))  # All tokens are attended to
    labels = torch.randint(0, 2, (batch_size,))  # Random binary labels

    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    # Training step
    loss = model.training_step(batch, batch_idx=0)

    # Check loss is a scalar tensor
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.dim() == 0, "Loss should be a scalar tensor"
    assert loss.item() >= 0, "Loss should be non-negative"
