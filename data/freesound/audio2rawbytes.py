
import numpy as np
import os
import sys


import essentia.standard as es


def save_as_mmap(waveform, mmap_path):
    # save as float16
    waveform = waveform.astype(np.float16)
    mmap_file = np.memmap(mmap_path, dtype=np.float16, mode='w+', shape=waveform.shape)
    mmap_file[:] = waveform[:]
    mmap_file.flush()


def process_audio_files(input_filename, output_sample_rate=16000):
    if input_filename.endswith((".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".wav",  ".mp4")): # ".aup", ".rpp" ".wv",
        print(f"Processing {input_filename}")

        mmap_path = input_filename.replace("/projects/", "/scratch/") + ".mmap"

        if os.path.exists(mmap_path):
            print(f"Skipping {mmap_path}, already exists.")
            exit()

        os.makedirs(os.path.dirname(mmap_path), exist_ok=True)

        waveform = es.MonoLoader(filename=input_filename, sampleRate=output_sample_rate, resampleQuality=4)()

        save_as_mmap(waveform, mmap_path)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_audio.py <file>")
        sys.exit(1)

    process_audio_files(sys.argv[1])
