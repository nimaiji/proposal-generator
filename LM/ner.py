import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def ner_from_file(file_path):
    # Read the file
    with open(file_path, "r") as file:
        text = file.read()

    # Process the text with SpaCy
    doc = nlp(text)

    # Initialize a dictionary to count the occurrences of each ORG entity
    org_counts = {}

    # Iterate through the entities and count the occurrences of ORG entities
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org = ent.text
            if org in org_counts:
                org_counts[org] += 1
            else:
                org_counts[org] = 1

    # Sort the dictionary by count in descending order
    sorted_org_counts = sorted(org_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the top 5 ORG entities
    print("Top 5 ORG entities:")
    for org, count in sorted_org_counts[:5]:
        print(f"{org}: {count}")

# Example usage
file_path = "/Users/nima/Desktop/A.I./Crawler/raw/https___easymantax.com_.txt"
ner_from_file(file_path)
