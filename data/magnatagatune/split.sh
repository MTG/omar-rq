#!/bin/bash

output_dir="data/magnatagatune"

all="$output_dir/all.txt"

# Set the directory
directory="/gpfs/projects/upf97/downstream_datasets/magnatagatune/"
find "$directory" -type f -name "*.mp3" > $all
count=$(wc -l < $all)
echo "Total number of files: $count"

train="$output_dir/train.txt"
validation="$output_dir/validation.txt"
test="$output_dir/test.txt"

: > $train
: > $validation
: > $test

# Iterate over the files
while IFS= read -r file; do
    # Get the directory name
    dir=$(dirname "$file")

    # Get the audio basename without the extension
    audio=$(basename "$file" .mp3)

    # Extract the last directory in the path
    last_dir=$(basename "$dir")

    # Check if last_dir starts with a digit 0-9 or is 'a' (assuming the first character)
    if [[ "$last_dir" =~ ^[0-9] ]] || [[ "$last_dir" == "a" ]]; then
        echo "$audio" >> $train

    # Check if last_dir starts with b or c (assuming the first character)
    elif [[ "$last_dir" =~ ^[b-c] ]]; then
        echo "$audio" >> $validation

    # Check if last_dir starts with d, e, or f (assuming the first character)
    elif [[ "$last_dir" =~ ^[d-f] ]]; then
        echo "$audio" >> $test

    else
        echo "$file does not match any expected category."
    fi

done < $all

# Count the number of files in each split
train_count=$(wc -l < $train)
validation_count=$(wc -l < $validation)
test_count=$(wc -l < $test)

echo "Number of files in train.txt: $train_count"
echo "Number of files in validation.txt: $validation_count"
echo "Number of files in test.txt: $test_count"

echo Done!