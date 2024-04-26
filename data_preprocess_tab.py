import streamlit as st
from data_processing import tokenize_and_process_data

def show_data_preprocess_tab():
    st.header("Data Pre-process")

    # Radio button to select the format
    selected_format = st.radio("Select Format:", ["CSV", "Json"])

    # Checkboxes for additional processing options
    remove_stop_words_option = st.checkbox("Remove Stop Words")
    remove_punctuation_option = st.checkbox("Remove Punctuation")
    lemmatization_option = st.checkbox("Lemmatization")

    # Button to process the data
    if st.button("Process the Data"):
        tokenize_and_process_data(selected_format, remove_stop_words_option, remove_punctuation_option, lemmatization_option)

        # Additional section to show statistics
        show_statistics()

def show_statistics():
    # Placeholder for statistics
    st.subheader("Statistics on Processed Data")
    st.warning("Processed data not available yet. Please run the data processing first.")
