#!/bin/bash

set -e
source /data0/palonso/ssl-mtg/envs/mtg-bsc/bin/activate


for id in \
    8bi35b82  `# Multi-view large encodec models` \
    msesipur i2h5dqb8  `# Freesound models` \
    ky3d2yu3 8jvszrnh  `# Other Variants by Oguz` \
    po9qoipi ga09fbmo  `# oguz cqt variants` \
    c4urat3s `# mel (small)` \
    y4thvf5f `# mel tiny (tiny)` \
    8avrux47 `# cqt` \
    molbhb3i qmmnknpy  `# encodec` \
    adlpqsh3 6a8dzz68 lfc02r16 9sn3yi5h izet8ved bm23z5le bvq3pd9u  `# Multi-view small encodec models` \
    3qttsr4s 4299e12g  `# Multi-view audio models (100K steps)` \
    hgu9kgyl  `# Multi-view audio models (200K steps)` \
    z6opk2rz mbgq9od4 ldtuk0yo yc10xacz 8bi35b82  `# Multi-view large encodec models` \
    msesipur i2h5dqb8  `# Freesound models` \

do
    logs_dir=/data0/palonso/ssl-mtg/weights/${id}
    cfg_file=$(ls ${logs_dir}/checkpoints/*.gin)

    echo ${cfg_file}

    CUDA_VISIBLE_DEVICES=0 python src/predict.py ${cfg_file} cfg/downstream/nynth_pitch.gin

    CUDA_VISIBLE_DEVICES=0 python src/downstream.py ${id} cfg/downstream/nsynth_pitch.gin
done
