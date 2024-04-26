# crawler_tab.py

import streamlit as st
import subprocess
from threading import Thread, Event
import time

# Global variables
crawler_thread = None
stop_crawling_event = Event()

# Function to run the web crawler script with the provided URL and depth in a separate thread
def run_crawler(url, depth):
    global crawler_thread, stop_crawling_event

    # Reset the event before starting
    stop_crawling_event.clear()

    # Pass the depth parameter to the crawler script
    crawler_thread = Thread(target=subprocess.run, args=(['python3', 'crawler.py', '--url', url, '--depth', str(depth)],), kwargs={'check': True})
    crawler_thread.start()

# Function to stop crawling
def stop_crawling_func():
    global crawler_thread, stop_crawling_event
    stop_crawling_event.set()

    if crawler_thread:
        crawler_thread.join()  # Wait for the crawler thread to finish

def show_crawler_tab():
    st.header("Web Crawler")

    # Input fields for the URL and depth
    url = st.text_input("Enter the URL to crawl:", "https://example.com")
    depth = st.slider("Select the depth for crawling:", min_value=1, max_value=10, value=1)

    # Button to start the crawler with the provided URL and depth
    if st.button("Start Web Crawler"):
        # Run the web crawler in a separate thread with the provided URL and depth
        run_crawler(url, depth)

    # Button to stop crawling
    stop_button = st.button("Stop", on_click=stop_crawling_func, key="stop_button")
    if stop_button:
        stop_crawling_func()

    # Toggle component to show/hide logs
    show_logs = st.checkbox("Show Logs")

    # Container for logs
    logs_container = st.empty()

    # Counter for the number of links scraped
    link_count = 0
    # Display the number of links scraped
    widget = st.empty()
    
    # If logs should be shown, read and display them
    if show_logs:
        try:
            while not stop_crawling_event.is_set():
                with open('crawler.log', 'r') as log_file:
                    logs = log_file.readlines()

                    for log in logs:
                        # Increment link count if a link is scraped
                        if "Scraping link:" in log:
                            link_count += 1

                        log_level = log.split(' - ')[1].strip().lower()

                        if log_level == 'info':
                            logs_container.write(f":green[{log}]")
                        elif log_level == 'warning':
                            logs_container.write(f":yellow[{log}]")
                        elif log_level == 'error':
                            logs_container.write(f":red[{log}]")
                        else:
                            logs_container.write(log)
                widget.write(f"Total Links Scraped: {link_count}")
                # Pause for a moment before checking for updates again
                time.sleep(1)

        except FileNotFoundError:
            st.warning("Log file not found. Run the crawler to generate logs.")
