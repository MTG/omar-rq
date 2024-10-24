#!/usr/bin/env bash

find /gpfs/projects/upf97/freesound/sounds/ -name "*" -type f | sed 's/^/python3 audio2rawbytes.py /' > all_audios.sh
split -l 100000 --numeric-suffixes=1 all_audios.sh all_audios.sh.task
chmod +x all_audios.sh*
