# home_tab.py

import streamlit as st

def show_home_tab():
    st.write("The Proposal Generator App leverages advanced technology, including web crawling and natural"
             " language processing (NLP) with BERT, to automate the proposal creation process for businesses."
             " By crawling target company websites, 🕸️ the app collects data on their services, which is then"
             " meticulously extracted using a pre-trained BERT model. 💡 Finally, based on the user's company"
             " information and the extracted service data, the app dynamically generates tailored proposals,"
             " streamlining the business development process. 💼✨ With its seamless integration of technology"
             " and business needs, the Proposal Generator App revolutionizes proposal creation, saving time and"
             " enhancing efficiency for companies seeking to win over potential clients.")

    st.header("Step 1: Crawler")
    st.write("The first step of the Proposal Generator App involves developing a web crawler to systematically explore "
             "the target company's website. This crawler collects valuable information regarding the services offered by "
             "the target company, extracting details such as service descriptions, titles, and any other pertinent data."
             " Utilizing libraries like Scrapy or Beautiful Soup, the crawler navigates through the website's structure,"
             " ensuring comprehensive coverage of all relevant service pages.")

    st.header("Step 2: Service Extraction")
    st.write("In the Service Extraction phase, a sophisticated natural language processing (NLP) technique is employed "
             "to accurately capture and categorize the services gathered by the crawler. Leveraging a pre-trained BERT "
             "model, the application fine-tunes its understanding of service-related text to ensure precise extraction. "
             "This involves processing the crawled data through the BERT model, which distills key service features and "
             "descriptions. The extracted information is then formatted and saved into easily accessible Excel (XLSX) "
             "and pickle (PKL) files for subsequent use in proposal generation.")

    st.header("Step 3: Proposal Generation")
    st.write("The Proposal Generation module is the heart of the application, where customized proposals are dynamically"
             " created based on user input and the extracted service data. By loading the preprocessed service "
             "information and incorporating user-provided details about both their own company and the target company, "
             "the app generates tailored proposals. These proposals include sections such as introductions, company "
             "overviews, proposed services, pricing structures, and contact information. Offering flexibility for "
             "customization, users can adjust the proposal content and format to suit their specific needs and preferences.")

    st.header("Optional: Data Preprocessing")
    st.write("The Data Preprocessing step, although optional, plays a crucial role in enhancing the quality and "
             "accuracy of the extracted service data. Through various preprocessing techniques, such as text "
             "normalization, entity recognition, and spell checking, the application ensures that the extracted "
             "information is clean and relevant. By eliminating noise and irrelevant details, the preprocessed data "
             "contributes to more accurate proposal generation, improving the overall performance and user experience "
             "of the Proposal Generator App.")
