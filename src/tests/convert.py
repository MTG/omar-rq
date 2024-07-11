

import os
import subprocess

# Define the base directory
base_directory = "/home/pedro/PycharmProjects/ssl-mtg/data/gtzan"

# Walk through all subdirectories and files
for root, _, files in os.walk(base_directory):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            temp_file_path = file_path + ".temp.wav"

            # Convert the file using ffmpeg and save to a temporary file
            subprocess.run(["ffmpeg", "-i", file_path, "-c:a", "pcm_s16be", temp_file_path, "-y"])

            # Replace the original file with the temporary file
            os.replace(temp_file_path, file_path)
            print(f"Converted {file_path}")