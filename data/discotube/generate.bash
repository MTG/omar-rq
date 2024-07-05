#!/bin/bash

# Set the directory
directory="/gpfs/projects/upf97/discotube"

# Find all mp3 files and store their absolute paths in an array
mapfile -t mp3_files < <(find "$directory" -type f -name "*.mp3" -print0 | xargs -0 realpath)

# Get the total number of mp3 files
total_files=${#mp3_files[@]}

# Calculate the number of files for train (80%) and test (20%)
train_count=$((total_files * 80 / 100))
test_count=$((total_files - train_count))

# Write the paths to the train.txt file and the test.txt file
> train.txt  # Clear the train.txt file if it exists
> test.txt   # Clear the test.txt file if it exists

for (( i=0; i<total_files; i++ )); do
  if [ $i -lt $train_count ]; then
    echo "${mp3_files[$i]}" >> train.txt
  else
    echo "${mp3_files[$i]}" >> test.txt
  fi
done

echo "Files have been written to 'train.txt' and 'test.txt'."
