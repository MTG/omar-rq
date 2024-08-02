import os
import numpy as np
from tqdm import tqdm

# Define the directory to search
directory = "/gpfs/scratch/upf97/mmaps"

# Define the size limit in bytes (0.1 MB)
size_limit = 0.1 * 1024 * 1024

# List to store files to be checked
files_to_check = []

# Walk through the directory and gather .mmap files
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".mmap"):
            file_path = os.path.join(root, file)
            files_to_check.append(file_path)

# Check each file size and remove if it is empty using numpy's memmap and less than the size limit
for file_path in tqdm(files_to_check, desc="Processing files"):
    try:
        file_size = os.path.getsize(file_path)
        if file_size < size_limit:
            # Load the file using numpy's memmap with float16 dtype
            mmap_array = np.memmap(file_path, dtype='float16', mode='r')
            del mmap_array  # Close the memmap
    except Exception as e:
        os.remove(file_path)
        print(f"Error processing file {file_path}: {e}")

