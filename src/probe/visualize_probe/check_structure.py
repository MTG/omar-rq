import os

import torch


def check_random_file():

    # get random path from ./embedding_structure
    random_path = os.path.join(os.getcwd(), 'embedding_structure')
    random_file = os.listdir(random_path)[0]

    data = torch.load(os.path.join(random_path, random_file))

    embedding = data['embedding']
    print("Embedding shape: ", embedding.shape)
    logits_frames = data['logits_frames']
    print("Logits frames shape: ", logits_frames.shape)
    frames_true = data['y_true']
    print("Frames true shape: ", frames_true.shape)
    logits_boundaries = data['logits_boundaries']
    print("Logits boundaries shape: ", logits_boundaries.shape)
    boundaries_true = data['boundaries_intervals']
    print("Boundaries true shape: ", boundaries_true.shape)


if __name__ == '__main__':
    check_random_file()