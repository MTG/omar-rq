import os

# Define the mapping function
def conversion(label):
    # code borrowed from algo1 figure (https://arxiv.org/pdf/2205.14700)
    substrings = [
        ("silence", "silence"),
        ("pre-chorus", "verse"),
        ("prechorus", "verse"),
        ("refrain", "chorus"),
        ("chorus", "chorus"),
        ("theme", "chorus"),
        ("stutter", "chorus"),
        ("verse", "verse"),
        ("rap", "verse"),
        ("section", "verse"),
        ("slow", "verse"),
        ("build", "verse"),
        ("dialog", "verse"),
        ("intro", "intro"),
        ("fadein", "intro"),
        ("opening", "intro"),
        ("bridge", "bridge"),
        ("trans", "bridge"),
        ("out", "outro"),
        ("coda", "outro"),
        ("ending", "outro"),
        ("break", "inst"),
        ("inst", "inst"),
        ("interlude", "inst"),
        ("impro", "inst"),
        ("solo", "inst")
    ]
    if label == "end":
        return "end"
    for s1, s2 in substrings:
        if s1 in label.lower():
            return s2
    return "inst"

# Define the function to normalize and save files
def normalize_segments(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    time, label = line.split(maxsplit=1)
                    normalized_label = conversion(label.strip())
                    outfile.write(f"{time} {normalized_label}\n")

# Paths to input and output folders
input_folder = 'segments'
output_folder = 'segments_norm'

# Normalize and save files
normalize_segments(input_folder, output_folder)
