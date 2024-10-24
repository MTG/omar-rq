import torchaudio
import torch
import numpy as np
import os
import sys

from tqdm import tqdm


import essentia.standard as es


def save_as_mmap(waveform, mmap_path):
    # save as float16
    waveform = waveform.astype(np.float16)
    mmap_file = np.memmap(mmap_path, dtype=np.float16, mode='w+', shape=waveform.shape)
    mmap_file[:] = waveform[:]
    mmap_file.flush()


def process_audio_files(input_directory, output_directory, output_sample_rate=16000):
    for file in os.listdir(input_directory):
        if file.endswith((".aif", ".aiff", ".aup", ".flac", ".m4a", ".mp3", ".ogg", ".rpp", ".wav", ".wv", ".mp4")):
            mp4_path = os.path.join(input_directory, file)

            # Create the corresponding path in the output directory
            relative_path = os.path.relpath(mp4_path, input_directory)
            mmap_path = os.path.join(output_directory, relative_path.replace(".mp4", ".mmap"))

            # Skip if the mmap file already exists
            if os.path.exists(mmap_path):
                print(f"Skipping {mmap_path}, already exists.")
                continue

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(mmap_path), exist_ok=True)

            # Load audio
            waveform = es.MonoLoader(filename=mp4_path, sampleRate=output_sample_rate, resampleQuality=4)()

            # Save downsampled audio as a memory-mapped file
            save_as_mmap(waveform, mmap_path)

            #print(f"Processed {mp4_path} and saved as {mmap_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_audio.py <subdir> <2020boolean>")
        sys.exit(1)

    subdir = sys.argv[1]
    input_directory = f"/gpfs/projects/upf97/freesound/sounds/{subdir}"
    output_directory = f"/gpfs/scratch/upf97/freesound/sounds/{subdir}"

    process_audio_files(input_directory, output_directory)
