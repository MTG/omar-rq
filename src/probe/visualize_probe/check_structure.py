import os
from random import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from scipy.signal import find_peaks

from src.probe.modules.structure_probe import peak_picking


def check_random_file():
    # Obtener una ruta aleatoria de ./embedding_structure
    random_path = os.path.join(os.getcwd(), "embedding_structure")
    random_file = os.listdir(random_path)[int(random() * len(os.listdir(random_path)))]

    data = torch.load(os.path.join(random_path, random_file))

    embedding = data["embedding"]  # Shape: (10, 469, 512)
    logits_frames = data["logits_frames"]  # Shape: (4690, 7)
    logits_frames = torch.sigmoid(logits_frames)  # Normalizar logits
    frames_true = data["y_true"]  # Shape: (4690,)
    logits_boundaries = data["logits_boundaries"]  # Shape: (4690,)
    logits_boundaries = torch.sigmoid(logits_boundaries)  # Normalizar logits boundaries

    print("Embedding shape: ", embedding.shape)
    print("Logits frames shape: ", logits_frames.shape)
    print("Frames true shape: ", frames_true.shape)
    print("Logits boundaries shape: ", logits_boundaries.shape)

    # Convertir tensores a numpy para graficar
    logits_frames_np = logits_frames.cpu().detach().numpy()
    frames_true_np = frames_true.cpu().detach().numpy()
    logits_boundaries_np = logits_boundaries.cpu().detach().numpy()

    # Obtener los peaks de logits_boundaries con threshold 0.064*3
    step_size = 0.064 * 3
    peaks = np.array(
        [
            pp
            for pp, _ in peak_picking(torch.from_numpy(logits_boundaries_np), step_size)
            if logits_boundaries_np[pp] > 0.2
        ]
    )

    # Crear una figura con 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), constrained_layout=True)

    # Definir los colores para los segmentos (usados en el subplot 1 y 2)
    num_segments = 8
    segment_colors = plt.cm.rainbow(np.linspace(0, 1, num_segments))

    # Etiquetas de las clases
    label_to_number = {
        "intro": 0,
        "verse": 1,
        "chorus": 2,
        "bridge": 3,
        "outro": 4,
        "inst": 5,
        "silence": 6,
    }
    number_to_label = {v: k for k, v in label_to_number.items()}

    # Ajustar la escala del eje x para que los subplots tengan el mismo rango de x
    num_frames = logits_frames_np.shape[0]  # Ejemplo: 4690 frames

    # 1er subplot: Segmentar y rellenar los segmentos de 0-7 con colores, ajustando el eje x
    for i in range(num_segments):
        segment_label = number_to_label.get(i, f"Segment {i}")  # Obtener etiqueta
        axs[0].fill_between(
            np.linspace(0, num_frames, num_frames),
            (frames_true_np == i).astype(int),
            color=segment_colors[i],
            label=segment_label,
        )
    axs[0].set_title("Segments with Labels (Colored)")
    axs[0].legend()
    axs[0].set_xlabel("Frames")
    axs[0].set_ylabel("Segment Presence")

    # 2do subplot: 7 curvas de logits_frames, ajustando el eje x
    for i in range(logits_frames_np.shape[1]):
        segment_label = number_to_label.get(i, f"Curve {i + 1}")  # Obtener etiqueta
        axs[1].plot(
            np.linspace(0, num_frames, num_frames),
            logits_frames_np[:, i],
            color=segment_colors[i],
            label=segment_label,
        )
    axs[1].set_title("Logits Frames Curves (Labeled)")
    axs[1].legend()
    axs[1].set_xlabel("Frames")
    axs[1].set_ylabel("Logits")

    # 3er subplot: Graficar una única curva (por ejemplo, la primera en logits_frames), ajustando el eje x
    axs[2].plot(
        np.linspace(0, num_frames, num_frames),
        logits_boundaries_np,
        color="blue",
        label=number_to_label[0],
    )
    axs[2].set_title(f"Single Curve ({number_to_label[0]})")
    axs[2].set_xlabel("Frames")
    axs[2].set_ylabel("Logits")

    # 4to subplot: Graficar logits_boundaries con peaks
    axs[3].plot(
        np.linspace(0, num_frames, num_frames),
        logits_boundaries_np,
        color="orange",
        label="Logits Boundaries",
    )
    axs[3].scatter(
        peaks, logits_boundaries_np[peaks], color="red", label="Peaks", zorder=5
    )
    axs[3].set_title("Logits Boundaries with Peaks")
    axs[3].legend()
    axs[3].set_xlabel("Frames")
    axs[3].set_ylabel("Logits Boundaries")

    # Ajustar el diseño para evitar solapamiento
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check_random_file()
