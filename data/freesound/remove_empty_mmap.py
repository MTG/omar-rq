import os
import numpy as np
from tqdm import tqdm

# Define the directory to search
directory = "/gpfs/scratch/upf97/freesound/sounds/"

# Define the size limit in bytes (0.1 MB)
size_limit = 0.05 * 1024 * 1024

# Lists to store file paths for logging
small_files = []
files_with_nans = []

# Walk through the directory and gather .mmap files
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".mmap"):
            file_path = os.path.join(root, file)
            small_files.append(file_path)

# Check each file size, and if small, look for NaNs using numpy's memmap
with open("small_files.txt", 'w') as small_file_log, open("files_with_nans.txt", 'w') as nan_file_log:
    for file_path in tqdm(small_files, desc="Checking files"):
        try:
            file_size = os.path.getsize(file_path)
            if file_size < size_limit:
                # Load the file using numpy's memmap with float16 dtype
                mmap_array = np.memmap(file_path, dtype='float16', mode='r')

                # Check for NaNs
                if np.isnan(mmap_array).any():
                    files_with_nans.append(file_path)
                    nan_file_log.write(f"{file_path}\n")
                    print(f"File {file_path} contains NaNs:", mmap_array[np.isnan(mmap_array)])

                # Log all small files
                small_file_log.write(f"{file_path}\n")

                del mmap_array  # Close the memmap

        except Exception as e:
            print(f"Error with file {file_path}: {e}")

# Display counts for the logs
print("Number of small files:", len(small_files))
print("Number of files with NaNs:", len(files_with_nans))
