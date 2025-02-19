# IMDB Sentiment Analysis with DistilBERT

## Description

This project fine-tunes a pre-trained DistilBERT model to perform sentiment analysis on movie reviews from the IMDB dataset. The IMDB dataset contains movie reviews labeled as either negative or positive. By training on this data, the model learns to accurately classify new reviews based on their sentiment.

## Project Structure

- **main_llm.py**  
  This script loads a subset of the IMDB dataset, tokenizes the text, and fine-tunes DistilBERT for binary sentiment classification. After training, the model and tokenizer are saved for later use.

- **inference.py**  
  This script loads the saved model and tokenizer and performs sentiment inference on new text inputs.

- **README.md**  
  This file provides an overview of the project, installation instructions, usage details, and other relevant information.

## Installation

- **Python Version:**  
  Use a compatible Python version (ideally between 3.7 and 3.9) for optimal compatibility.

- **Environment Setup:**  
  Create and activate a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

- **Required Packages:**  
  Install the necessary libraries such as Hugging Face Transformers, Datasets, Evaluate, and PyTorch. Additional packages may be required for extended functionality.
  ```bash
  pip3 install -r requirements.txt
  ```

## Usage

1. **Training:**  
   Run the main_llm.py script to fine-tune DistilBERT on a subset of the IMDB dataset. This script will:
- Load and shuffle the IMDB dataset.
- Tokenize the reviews.
- Fine-tune the model for binary sentiment classification.
- Save the fine-tuned model and tokenizer to the ./saved_model directory.

    ```Bash
    python3 main.py
    ```

2. **Inference:**  
   After training, run the inference script to load the saved model and perform sentiment analysis on new movie reviews. The model outputs a sentiment prediction (either positive or negative) based on the input text.
    ```Bash
    python3 inference.py
    ```

## Key Points

- **Supervised Learning:**  
  The project uses supervised training by providing both the movie review texts (features) and their corresponding sentiment labels (targets).

- **IMDB Dataset:**  
  The dataset consists of 50,000 movie reviews (25,000 for training and 25,000 for testing), though a smaller subset is used in this example for demonstration purposes.

    Example from the dataset, here label "0" denotes a negative sentiment.
    ```Bash
    {
        "text": "I rented I AM CURIOUS-YELLOW...But really, this film doesn\"t have much of a plot.",
        "label": 0
    }
    ```

- **Binary Sentiment Classification:**  
  The model is configured to classify reviews into two categories (negative and positive) by setting the classification head to output two logits.

- **Fine-Tuning Process:**  
  The pre-trained DistilBERT model is adapted to recognize sentiment patterns in text by training on the labeled IMDB reviews.

## Contributing

Contributions are welcome! Feel free to explore the code, make improvements, or suggest new features. If you wish to contribute, please follow the repository’s contribution guidelines and open an issue or submit a pull request.

## License

This project is open source. Please include any licensing details if applicable.

## Contact

For any questions or issues, please open an issue in the repository or contact the project maintainer.
