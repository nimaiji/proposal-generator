import streamlit as st
from home_tab import show_home_tab
from crawler_tab import show_crawler_tab
from data_preprocess_tab import show_data_preprocess_tab
from service_extraction_tab import show_service_extraction_tab
from proposal_generation_tab import show_proposal_generation_tab

def main():
    st.title("Proposal Generator App")

    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Home"

    home_tab_button = st.sidebar.button("Home")
    crawler_tab_button = st.sidebar.button("Web Crawler")
    data_preprocess_tab_button = st.sidebar.button("Data Pre-process")
    service_extraction_tab_button = st.sidebar.button("Service Extraction")
    proposal_generation_tab_button = st.sidebar.button("Proposal Generation")

    if home_tab_button:
        st.session_state.selected_tab = "Home"

    elif crawler_tab_button:
        st.session_state.selected_tab = "Web Crawler"

    elif data_preprocess_tab_button:
        st.session_state.selected_tab = "Data Pre-process"
        
    elif service_extraction_tab_button:
        st.session_state.selected_tab = "Service Extraction"

    elif proposal_generation_tab_button:
        st.session_state.selected_tab = "Proposal Generation"

    if st.session_state.selected_tab == "Home":
        show_home_tab()

    elif st.session_state.selected_tab == "Web Crawler":
        show_crawler_tab()

    elif st.session_state.selected_tab == "Data Pre-process":
        show_data_preprocess_tab()
        
    elif st.session_state.selected_tab == "Service Extraction":
        show_service_extraction_tab()
        
    elif st.session_state.selected_tab == "Proposal Generation":
        show_proposal_generation_tab()

if __name__ == "__main__":
    main()
