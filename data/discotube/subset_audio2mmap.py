import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


# Function to process a subdir with the given mode
def process_subdir(args):
    subdir, mode = args
    print(subdir)
    subprocess.run(["python3", "audio2rawbytes.py", subdir, str(mode)])


# Function to get subdirectories up to a limit
def get_subdirs(base_dir):
    subdirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    return subdirs


# Directories to process with a limit of 5000 each to not exceed 10000 total
dirs_2020_09 = get_subdirs("/gpfs/projects/upf97/discotube/discotube-2020-09/audio/")
dirs_2023_03 = get_subdirs(
    "/gpfs/projects/upf97/discotube/discotube-2023-03/audio-new/audio/"
)
print(dirs_2020_09)

# Combine all tasks
tasks = [(subdir, 0) for subdir in dirs_2020_09] + [
    (subdir, 1) for subdir in dirs_2023_03
]

len_tasks = len(tasks)
block_len = len_tasks // 40
subset = int(sys.argv[1])
subset_tasks = tasks[subset * block_len : (subset + 1) * block_len]
for task in tqdm(subset_tasks):
    process_subdir(task)
