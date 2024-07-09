

if __name__ == '__main__':
    import torchaudio
    x, sr = torchaudio.load("/home/pedro/PycharmProjects/ssl-mtg/data/gtzan/blues/blues.00032.wav")
    print(x)