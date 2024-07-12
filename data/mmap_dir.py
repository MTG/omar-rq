import torchaudio
import torch
import numpy as np
import os
import sys

from tqdm import tqdm


def load_audio(mp4_path):
    waveform, sample_rate = torchaudio.load(mp4_path, format='mp4')
    return waveform, sample_rate


def transform_audio(waveform, resampler,):
    downsampled_waveform = resampler(waveform)
    waveform_mono = torch.mean(downsampled_waveform, dim=0, keepdim=True)
    return waveform_mono


def save_as_mmap(waveform, mmap_path):
    np_waveform = waveform.numpy().astype(np.float16)  # Convert to float16
    mmap_file = np.memmap(mmap_path, dtype=np.float16, mode='w+', shape=np_waveform.shape)
    mmap_file[:] = np_waveform[:]
    mmap_file.flush()


def process_audio_files(input_directory, output_directory, output_sample_rate=16000):
    resampler = None  # Declare resampler here to use it for all files
    for file in tqdm(os.listdir(input_directory)):
        if file.endswith(".mp4"):
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
            waveform, sample_rate = load_audio(mp4_path)

            # Initialize the resampler only once with the original sample rate of the first file
            if resampler is None:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=output_sample_rate)

            # Downsample audio
            downsampled_waveform = transform_audio(waveform, resampler)

            # Save downsampled audio as a memory-mapped file
            save_as_mmap(downsampled_waveform, mmap_path)

            print(f"Processed {mp4_path} and saved as {mmap_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_audio.py <subdir> <2020boolean>")
        sys.exit(1)

    subdir = sys.argv[1]
    is_2020 = sys.argv[2]
    # if 0 2020 if 1 2023
    metadir = "discotube-2020-09" if is_2020 == "0" else "discotube-2023-03"
    input_directory = f"/gpfs/projects/upf97/discotube/{metadir}/audio-new/audio/{subdir}"
    output_directory = f"/gpfs/projects/upf97/discotube/{metadir}/mmaps/{subdir}"

    process_audio_files(input_directory, output_directory)

