from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
import json
from sklearn.metrics import classification_report

MODEL_DIR = "./fine_tuned_model"
TEST_JSON = "data_test.json"

def load_test_data(test_json_path, label_mapping_path):
    with open(test_json_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Load label mapping
    with open(label_mapping_path, "r", encoding="utf-8") as label_file:
        int_to_label = json.load(label_file)
        label_to_int = {v: int(k) for k, v in int_to_label.items()}

    valid_labels = set(label_to_int.keys())
    test_data = [
        {"input": item["code_snippet"], "label": label_to_int[item["vulnerability_type"]]}
        for item in data if item["vulnerability_type"] in valid_labels
    ]
    
    dataset = Dataset.from_dict({
        "input": [item["input"] for item in test_data],
        "label": [item["label"] for item in test_data]
    })
    return dataset, int_to_label

def evaluate_model(model_dir, test_json_path):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the test dataset and label mapping
    test_dataset, int_to_label = load_test_data(test_json_path, f"{model_dir}/label_mapping.json")
    
    # Tokenize the test data
    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)

    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model
    all_labels = []
    all_preds = []

    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    # Generate classification report
    labels = [int_to_label[str(i)] for i in sorted(int_to_label.keys())]
    print(classification_report(all_labels, all_preds, target_names=labels))

if __name__ == "__main__":
    evaluate_model(MODEL_DIR, TEST_JSON)
