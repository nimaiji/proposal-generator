import os

def keep_first_file_delete_others(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            files_in_dir = os.listdir(dir_path)
            if files_in_dir:
                # Keep the first file
                first_file = files_in_dir[0]
                for file_name in files_in_dir[1:]:
                    file_path = os.path.join(dir_path, file_name)
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path)

# Replace 'path_to_your_directory' with the path to the directory containing the folders you want to check
directory_path = './dataset'

keep_first_file_delete_others(directory_path)
