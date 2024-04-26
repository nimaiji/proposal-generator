import subprocess
import threading
import sys
import pandas as pd
import numpy as np

scrapped = 0
count = 0
percentage = 0

def run_external_script(script_path, url, name, industry, city):
    global scrapped
    global count
    global percentage
    command = ["python3", script_path, '--url', 'https://' + url, '--output_folder', f'./dataset/{industry}/{name}_{city}', '--depth', str(1), '--delete_previous', 'false']  
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode != 0:
        print(f"Error running script {script_path}:\n{error.decode('utf-8')}")
    else:
        scrapped += 1
        # Calculate and print the percentage of websites successfully scraped
        success_percentage = (scrapped / count) * 100
        if int(success_percentage*100) > int(percentage*100):
            percentage = success_percentage
            print(f"Percentage of websites scraped: {success_percentage:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main_script.py script_path dataset_path num_threads")
        sys.exit(1)

    script_path = sys.argv[1]
    dataset_path = sys.argv[2]
    num_threads = int(sys.argv[3])
    df = pd.read_csv(dataset_path)
    items = np.array(list(df.itertuples()))
    count = items.shape[0]
    
    threads = []
    for chunk in np.array_split(items, items.shape[0] / num_threads):
        for record in chunk:
            thread = threading.Thread(target=run_external_script, args=(script_path, record[3], record[2], record[4], record[8]))
            threads.append(thread)

    # Start the threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


