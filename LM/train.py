import torch
from transformers import BertTokenizer, BertForTokenClassification
import re

def extract_information(raw_text):
    # Load pre-trained BERT model and tokenizer
    model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)

    # Tokenize the raw text
    tokenized_text = tokenizer.tokenize(raw_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Run inference
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)

    # Extract labels
    predictions = torch.argmax(outputs[0], dim=2)
    labels = [prediction.item() for prediction in predictions[0]]

    # Convert label indices back to tokens
    predicted_tokens = tokenizer.convert_ids_to_tokens(labels)

    # Extract company name and services
    company_name = ''
    services = ''
    inside_entity = False
    for token, label in zip(tokenized_text, predicted_tokens):
        if label.startswith('B-ORG'):
            if inside_entity:
                services += ' ' + token
            else:
                company_name += ' ' + token
                inside_entity = True
        elif label.startswith('I-ORG'):
            company_name += token
        elif label.startswith('B-MISC'):
            if inside_entity:
                company_name += ' ' + token
            else:
                services += ' ' + token
                inside_entity = True
        elif label.startswith('I-MISC'):
            services += token
        else:
            inside_entity = False

    # Clean up company name and services
    company_name = re.sub(r'^\s+', '', company_name)
    services = re.sub(r'^\s+', '', services)

    return {'Name': company_name, 'Services': services}

# Example usage
raw_text = "Hi this is Napier University. We provide software development and consultancy services."
information = extract_information(raw_text)
print("Input:", raw_text)
print("Output:", information)
