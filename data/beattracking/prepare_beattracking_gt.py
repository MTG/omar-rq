import pickle as pk
import random
from pathlib import Path

output_dir = Path("../downstream_datasets/beattracking/")

ball_dir = output_dir / "BallroomAnnotations"
smcb_dir = output_dir / "SMC_MIREX/SMC_MIREX_Annotations_05_08_2014/"
gtza_dir = output_dir / "gtzan_tempo_beat/beats/"
hans_dir = output_dir / "beat_hainsworth/metadata/"

val_ratio = 0.2


def process_gtzan(filename: Path):
    stem = filename.stem

    _, genre, idx = stem.split("_")
    new_stem = f"{genre}.{idx}"

    return filename.with_stem(new_stem)


def get_id(path: Path):
    if "gtzan" in str(path):
        path = process_gtzan(path)
    stem = Path(path).stem

    return stem[:3] + "/" + stem


def process_tsvish(data_dir: Path):
    # get all files with .beats extention
    files = list(data_dir.rglob("*.beats"))
    if len(files) == 0:
        files = list(data_dir.rglob("*.txt"))

    print(f"Found {len(files)} files in {data_dir}")

    gt = dict()
    for filename in files:
        with open(filename, "r") as f:
            lines = f.readlines()
            timestamps = []
            for line in lines:
                values = line.strip().split("\t")
                if len(values) == 1:
                    values = values[0].split(" ")
                ts = values[0]
                timestamps.append(float(ts))

        gt[get_id(filename)] = timestamps

    # ../embeddings/8bi35b82/beattracking/roc/rock.00001.pt
    # ../downstream_datasets/beattracking/gtzan_tempo_beat/beats/gtzan_pop_00089.beats

    if "gtzan" in str(data_dir):
        files = [process_gtzan(f) for f in files]

    # ../downstream_datasets/beattracking/gtzan_tempo_beat/beats/rock.00001.beats

    print(f"Example filename: {list(gt.keys())[0]}")
    return gt, files


def process_beatlist(data_dir: Path):
    # get all files with .beats extention
    files = list(data_dir.rglob("*.beats"))
    print(f"Found {len(files)} files in {data_dir}")

    gt = dict()
    for filename in files:
        with open(filename, "r") as f:
            line = f.readlines()[0].strip()
            timestamps = line.split(" ")
            timestamps = [float(ts) for ts in timestamps]

        gt[get_id(filename)] = timestamps

    print(f"Example filename: {list(gt.keys())[0]}")
    return gt, files


ball_gt, files_ball = process_tsvish(ball_dir)
smcb_gt, files_smcb = process_tsvish(smcb_dir)
gtza_gt, files_gtza = process_tsvish(gtza_dir)
hans_gt, giles_hans = process_beatlist(hans_dir)


# merge all dicts
gt = {**ball_gt, **smcb_gt, **hans_gt, **gtza_gt}


# merge ball smcb, and hans filelists and shuffle
files = files_ball + files_smcb + giles_hans
random.shuffle(files)
# split into train, val sets
val_size = int(val_ratio * len(files))
train_files = files[val_size:]
val_files = files[:val_size]

# save the groundtruth
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "groundtruth.pkl", "wb") as f:
    pk.dump(gt, f)

# save the filelists
with open(output_dir / "filelist_train.txt", "w") as f:
    for file in train_files:
        f.write(str(file) + "\n")
with open(output_dir / "filelist_val.txt", "w") as f:
    for file in val_files:
        f.write(str(file) + "\n")

# save the test filelist with the gtza files
with open(output_dir / "filelist_test.txt", "w") as f:
    for file in files_gtza:
        f.write(str(file) + "\n")
