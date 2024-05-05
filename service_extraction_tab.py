import streamlit as st
import subprocess

def call_text_classification_script(folder_path, output_excel, output_pickle):
    """
    Calls the text classification script with the provided arguments.

    Args:
    folder_path (str): Path to the folder containing text files.
    output_excel (str): Path to save the output Excel file.
    output_pickle (str): Path to save the output Pickle file.
    """
    command = ["python", "./LM/inference.py", folder_path, output_excel, output_pickle]
    try:
        subprocess.run(command, check=True)
        st.write("Service extraction script executed successfully.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing service extraction script: {e}")

def show_service_extraction_tab():
    st.header("Service Extraction")
    st.write("This is the Service Extraction tab.")
    st.write("You can call your service extraction script here.")

    folder_path = st.text_input("Enter folder path:", "./raw")

    use_custom_output_excel = st.checkbox("Customize output Excel file path")
    output_excel = ""
    if use_custom_output_excel:
        output_excel = st.text_input("Enter output Excel file path:", "./output.xlsx")
    
    use_custom_output_pickle = st.checkbox("Customize output Pickle file path")
    output_pickle = ""
    if use_custom_output_pickle:
        output_pickle = st.text_input("Enter output Pickle file path:", "./output.pickle")

    if st.button("Run Service Extraction"):
        if folder_path:
            call_text_classification_script(folder_path, output_excel, output_pickle)
        else:
            st.error("Please provide the folder path.")

if __name__ == "__main__":
    show_service_extraction_tab()
