import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

MODEL_DIR = "./fine_tuned_model"
TEST_JSON = "test_data.json"

def load_test_data(test_json):
    """Load and preprocess test data from JSON."""
    with open(test_json, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    dataset = Dataset.from_dict({
        "input": [item["input"] for item in data],
        "label": [item["label"] for item in data]
    })
    return dataset

def evaluate_model(test_dataset, model, tokenizer):
    """Evaluate the model on the test dataset."""
    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)

    tokenized_test_data = test_dataset.map(tokenize_function, batched=True)
    unique_labels = set(test_dataset["label"])
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    tokenized_test_data = tokenized_test_data.map(lambda x: {"label": label_to_int[x["label"]]})
    tokenized_test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    dataloader = torch.utils.data.DataLoader(tokenized_test_data, batch_size=8)

    all_predictions = []
    all_labels = []

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "label"}
            labels = batch["label"].to(device)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    test_dataset = load_test_data(TEST_JSON)
    metrics = evaluate_model(test_dataset, model, tokenizer)

    print("Evaluation Results:")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
