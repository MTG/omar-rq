from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import essentia.standard as es


def save_as_mmap(waveform, mmap_path):
    """Save waveform as a memory-mapped file in float16 format."""
    waveform = waveform.astype(np.float16)
    mmap_file = np.memmap(
        mmap_path,
        dtype=np.float16,
        mode="w+",
        shape=waveform.shape,
    )
    mmap_file[:] = waveform[:]
    mmap_file.flush()

    # Ensure the file is properly closed
    del mmap_file


def process_audio_files(
    input_filelist: list,
    input_dir: Path,
    output_dir: Path,
    sample_rate: float,
):
    """Process audio files and save them as memory-mapped files."""
    for audio_path in input_filelist:
        try:
            # Create the corresponding path in the output directory
            rel_path = audio_path.relative_to(input_dir)
            audio_path = input_dir / rel_path
            mmap_path = output_dir / rel_path.with_suffix(".mmap")

            # Skip if the mmap file already exists
            if mmap_path.exists():
                print(f"Skipping {mmap_path}, already exists.")
                continue

            # Ensure the output directory exists
            mmap_path.parent.mkdir(parents=True, exist_ok=True)

            # Load audio
            waveform = es.MonoLoader(
                filename=str(audio_path),
                sampleRate=sample_rate,
                resampleQuality=4,
            )()

            save_as_mmap(waveform, mmap_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue


def audio2mmap(
    n_tasks: int,
    task_id: int,
    input_dir: Path,
    output_dir: Path,
    sample_rate: float,
):
    """Process audio files in subsets and save as memory-mapped files."""

    # List n_files with glob
    n_files = list(input_dir.rglob("*.*"))

    chunk_size = len(n_files) // n_tasks

    print(f"Total files: {len(n_files)}")
    print(f"Chunk size: {chunk_size}")
    print(f"processing task {task_id + 1}/{n_tasks}")

    if task_id == n_tasks - 1:
        chunk_files = n_files[task_id * chunk_size :]
    else:
        chunk_files = n_files[task_id * chunk_size : (task_id + 1) * chunk_size]

    process_audio_files(
        input_filelist=chunk_files,
        input_dir=input_dir,
        output_dir=output_dir,
        sample_rate=sample_rate,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Process audio files in subsets.")
    parser.add_argument(
        "--n-tasks",
        type=int,
        required=True,
        help="Number of subsets to process",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Index of the subset to process",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save processed files",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=16000,
        help="Sample rate for audio processing",
    )

    args = parser.parse_args()

    audio2mmap(
        n_tasks=args.n_tasks,
        task_id=args.task_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
    )
