import numpy as np
import soundfile as sf

if __name__ == '__main__':
    # Load the data
    audio = np.memmap("data/local/blues_mmap/blues.00000.mmap", dtype=np.float16, mode='r')
    print(audio.shape)
    # save as test.wav
    audio = audio.astype(np.float32).tolist()
    sf.write("data/test.wav", audio, 16000)
    print("Saved as test.wav")
