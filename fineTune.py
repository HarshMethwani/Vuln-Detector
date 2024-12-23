from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import json
from transformers import AutoTokenizer
import torch

MODEL_NAME = "microsoft/codebert-base"
OUTPUT_MODEL_DIR = "./fine_tuned_model"
OUTPUT_JSON = "fine_tune_data.json"

def fine_tune_model(tokenized_data_path, output_dir, model_name="microsoft/codebert-base"):
    with open(tokenized_data_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    dataset = Dataset.from_dict({
        "input": [item["code_snippet"] for item in data],
        "label": [item["vulnerability_type"] for item in data]
    })
    valid_labels = {"Re-entrancy", "Timestamp-Dependency"}
    dataset = dataset.filter(lambda x: x["label"] in valid_labels)

    label_to_int = {label: idx for idx, label in enumerate(valid_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    dataset = dataset.map(lambda x: {"label": label_to_int[x["label"]]})
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(valid_labels)).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
        no_cuda=not torch.cuda.is_available(),  # Automatically use GPU if available
        fp16=torch.cuda.is_available(),         # Use mixed precision if GPU supports it
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(f"{output_dir}/label_mapping.json", "w") as label_file:
        json.dump(int_to_label, label_file)

    print(f"Fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    fine_tune_model(OUTPUT_JSON, OUTPUT_MODEL_DIR)
