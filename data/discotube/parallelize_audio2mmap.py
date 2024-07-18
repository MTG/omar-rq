import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


# Function to process a subdir with the given mode
def process_subdir(args):
    subdir, mode = args
    print(subdir)
    subprocess.run(['python3', 'audio2rawbytes.py', subdir, str(mode)])

# Function to get subdirectories up to a limit
def get_limited_subdirs(base_dir, limit=10000000):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return subdirs[:limit]

# Directories to process with a limit of 5000 each to not exceed 10000 total
dirs_2020_09 = get_limited_subdirs('/gpfs/projects/upf97/discotube/discotube-2020-09/audio/', 50000)
dirs_2023_03 = get_limited_subdirs('/gpfs/projects/upf97/discotube/discotube-2023-03/audio-new/audio/', 50000)
print(dirs_2020_09)

# Combine all tasks
tasks = [(subdir, 0) for subdir in dirs_2020_09] + [(subdir, 1) for subdir in dirs_2023_03]

# Run tasks in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=75) as executor:
    list(tqdm(executor.map(process_subdir, tasks), total=len(tasks), desc="Processing Subdirectories"))