import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def main():
    model_name = "distilbert-base-uncased"
    
    # Load the IMDB dataset
    dataset = load_dataset("imdb")
    # print(dataset["train"][0])
    
    # Use a smaller subset for quick fine-tuning (for demonstration purposes)
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(500))
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenization function to process text
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    
    # Tokenize the datasets (batched for speed)
    tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = small_eval_dataset.map(tokenize_function, batched=True)
    
    # Set the format for PyTorch tensors
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )
    
    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    # Save the fine-tuned model and tokenizer so they can be loaded for inference
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Model and tokenizer saved to ./saved_model")

if __name__ == "__main__":
    main()
