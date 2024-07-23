# rename_files.py

import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith("000"):
            new_name = filename.replace("000", "")
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
            print(f"Renamed {filename} to {new_name}")

if __name__ == "__main__":
    data_dir = 'data/raw'
    rename_files(data_dir)
