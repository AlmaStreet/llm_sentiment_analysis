import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_sentiment(text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    # Get model outputs (logits)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Determine predicted label (0 or 1)
    predicted_label = torch.argmax(logits, dim=1).item()
    
    # Map the label to a sentiment string (assuming 0 = negative, 1 = positive)
    sentiment = "positive" if predicted_label == 1 else "negative"
    return sentiment

if __name__ == "__main__":
    # Load the saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./saved_model")
    model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
    
    # Example usage:
    sample_text = "this movie sucks!"
    sentiment = predict_sentiment(sample_text, tokenizer, model)
    print(f"Sentiment: {sentiment}")
