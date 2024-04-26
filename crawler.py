# crawler.py

import argparse
import logging
import os
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from colorlog import ColoredFormatter

def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    console_format = '%(log_color)s%(asctime)s - [%(levelname)s] - %(message)s'
    console_formatter = ColoredFormatter(console_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)

def delete_existing_data(output_folder):
    if os.path.exists(output_folder):
        for file_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file_name)
            os.remove(file_path)
        os.rmdir(output_folder)
        logging.warning(f"Existing '{output_folder}' folder deleted")

def remove_hidden_elements(element):
    # Remove element if it has display: none style
    
    if element.get('style') and 'display: none' in element.get('style') or (element.get('class') and 'elemntor-hidden-desktop' in element.get('class')):
        element.extract()
        return

    # Recursively remove hidden elements from children
    for child in element.find_all(recursive=False):
        remove_hidden_elements(child)


def crawl(url, depth, output_folder, delete_previous):
    if delete_previous == 'true':
        delete_existing_data(output_folder)
    
    base_url = urlparse(url).scheme + "://" + urlparse(url).netloc
    
    visited_links = set()

    def is_valid_address(address):
        return address.startswith(base_url)

    def scrape_page(url, current_depth):
        if current_depth > depth:
            return

        if url in visited_links:
            return

        visited_links.add(url)

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract links from the page
            links = soup.find_all('a', href=True)

            for link in links:
                absolute_link = urljoin(base_url, link.get('href'))

                # Check if the address starts with the defined URL
                if is_valid_address(absolute_link):
                    logging.info(f"Scraping: {absolute_link}")
                    page_content = ''
                    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'tr']):
                        if not tag.has_attr('href'):
                            for text in tag.stripped_strings:
                                if len(text) > 2 and len(text.split(' ')) > 8 and '.' in text and '©' not in text and 'http' not in text and 'www' not in text:
                                    page_content += text + '\n'
                    

                    file_name = f"{output_folder}/{absolute_link.replace('/', '_').replace(':', '_')}.txt"
                    if page_content != '':
                        with open(file_name, 'w', encoding='utf-8') as file:
                            file.write(page_content)

                    # Recursive call to scrape the next page
                    scrape_page(absolute_link, current_depth + 1)

        except Exception as e:
            logging.error(f"Error processing link {url}: {str(e)}")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Start crawling from the provided URL
    scrape_page(url, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Crawler")
    parser.add_argument("--url", type=str, help="URL to start crawling from", required=True)
    parser.add_argument("--depth", type=int, default=1, help="Maximum depth for crawling")
    parser.add_argument("--log_path", type=str, default="crawler.log", help="Path to the log file")
    parser.add_argument("--output_folder", type=str, default="raw", help="Folder to save raw page content")
    parser.add_argument("--delete_previous", type=str, default='true', help="Delete previous files")
    args = parser.parse_args()
    setup_logging(args.log_path)
    crawl(args.url, args.depth, args.output_folder, args.delete_previous)
    logging.info('Scraping finished!')
