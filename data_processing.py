import os
import glob
import pandas as pd
from collections import Counter
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_punctuation(text):
    words = word_tokenize(text)
    words_without_punct = [word for word in words if word.isalnum()]
    return ' '.join(words_without_punct)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def remove_not_word(text):
    words = word_tokenize(text)
    cleaned = [word for word in words if len(word) > 2]
    return ' '.join(cleaned)

def tokenize_and_process_data(choice, remove_stop_words_option, remove_punctuation_option, lemmatization_option):
    raw_folder_path = "raw"

    # Get all files in the raw folder
    files = glob.glob(os.path.join(raw_folder_path, "*"))

    # Tokenize and process data
    word_counts = Counter()

    # Initialize progress bar and text
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, file_path in enumerate(files, start=1):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().lower()
            content = remove_not_word(content)
            # Apply processing options
            if remove_stop_words_option:
                content = remove_stop_words(content)
            if remove_punctuation_option:
                content = remove_punctuation(content)
            if lemmatization_option:
                content = lemmatize_text(content)

            words = content.split()
            word_counts.update(words)

        # Update progress bar and text
        progress_percentage = int((i / len(files)) * 100)
        progress_bar.progress(progress_percentage)
        progress_text.text(f"Processing: {progress_percentage}%")

    total_words = sum(word_counts.values())
    st.write(f'Word total count: {total_words}')
    word_probabilities = {word: count / total_words for word, count in word_counts.items()}

    # Create a DataFrame
    data = {'Word': list(word_counts.keys()), 'Frequency': list(word_counts.values()), 'Probability': list(word_probabilities.values())}
    df = pd.DataFrame(data)

    # Export to CSV or JSON based on choice
    if choice == "CSV":
        df.to_csv("processed_data.csv", index=False)
    elif choice == "Json":
        df.to_json("processed_data.json", orient="records")

    st.success(f"Data processed and exported to {choice} successfully!")

    # Display a preview of the processed data
    st.subheader("Preview of Processed Data")
    st.dataframe(df.head())

    # Show statistics
    show_statistics(df)

def show_statistics(df):
    st.subheader("Statistics on Processed Data")

    # Highest frequency
    highest_frequency = df.loc[df['Frequency'].idxmax()]
    st.write(f"Highest Frequency: Word '{highest_frequency['Word']}' with Frequency {highest_frequency['Frequency']}")

    # Highest probability
    highest_probability = df.loc[df['Probability'].idxmax()]
    st.write(f"Highest Probability: Word '{highest_probability['Word']}' with Probability {highest_probability['Probability']:.4f}")

    # Lowest frequency
    lowest_frequency = df.loc[df['Frequency'].idxmin()]
    st.write(f"Lowest Frequency: Word '{lowest_frequency['Word']}' with Frequency {lowest_frequency['Frequency']}")

    # Lowest probability
    lowest_probability = df.loc[df['Probability'].idxmin()]
    st.write(f"Lowest Probability: Word '{lowest_probability['Word']}' with Probability {lowest_probability['Probability']:.4f}")

    # Top 5 words
    top_5_words = df.nlargest(5, 'Frequency')
    st.write("Top 5 Words:")
    st.dataframe(top_5_words[['Word', 'Frequency']])
