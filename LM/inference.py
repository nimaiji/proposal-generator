import os
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import argparse
import logging
from colorlog import ColoredFormatter

# Set up logging
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def predict_class(text, model, tokenizer):
    """
    Predicts the class of a given text using the provided model and tokenizer.

    Args:
    text (str): The input text to predict the class for.
    model (BertForSequenceClassification): The BERT model for sequence classification.
    tokenizer (BertTokenizer): The BERT tokenizer.

    Returns:
    tuple: A tuple containing the predicted class index and confidence score.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    predicted_class = torch.argmax(probs).item()
    return predicted_class, probs[predicted_class].item()

def main(folder_path, output_excel, output_pickle):
    """
    Main function to process text files in a folder, predict their classes, and save results.

    Args:
    folder_path (str): Path to the folder containing text files.
    output_excel (str): Path to save the output Excel file.
    output_pickle (str): Path to save the output Pickle file.
    """
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load("./LM/bert_model.pt", map_location=torch.device('cpu')))
    model.eval()

    total_texts = 0
    processed_texts = 0

    sentences_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
                total_texts += len(lines)  # Count total texts
                for line in lines:
                    text = line.strip()
                    predicted_class, confidence = predict_class(text, model, tokenizer)
                    if predicted_class not in sentences_dict:
                        sentences_dict[predicted_class] = set()  # Use set to store unique sentences
                    sentences_dict[predicted_class].add(text)  # Add sentence to set
                    processed_texts += 1  # Count processed texts
                    # Log progress
                    percentage = (processed_texts / total_texts) * 100
                    logger.info(f"Processed {processed_texts}/{total_texts} texts ({percentage:.2f}% done)")

    data = []
    for class_index, sentences in sentences_dict.items():
        for sentence in sentences:
            data.append({"Class": class_index, "Text": sentence})
    df = pd.DataFrame(data)

    df.to_excel(output_excel, index=False)
    df.to_pickle(output_pickle)

    # Add colored logs
    logger.info("Output Excel and Pickle files saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files to predict classes using BERT.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing text files.")
    parser.add_argument("output_excel", type=str, help="Path to save the output Excel file.")
    parser.add_argument("output_pickle", type=str, help="Path to save the output Pickle file.")
    args = parser.parse_args()

    main(args.folder_path, args.output_excel, args.output_pickle)
