#!/bin/bash

# Set the directory
directory="/gpfs/scratch/upf97/freesound/sounds/"
find "$directory" -type f -name "*.mmap" > fs_all.txt
# exclude the files in small_files.txt
grep -v -f small_files.txt fs_all.txt > fs_all_copy.txt
mv fs_all_copy.txt fs_all.txt

# exclude the files in nan_files.txt
grep -v -f files_with_nans.txt fs_all.txt > fs_all_copy.txt
mv fs_all_copy.txt fs_all.txt

count=$(wc -l < fs_all.txt)
echo "Total number of files: $count"

count_train=$(echo "scale=0; $count * 0.8" | bc)
# to integer
count_train=${count_train%.*}
count_test=$(echo "scale=0; $count * 0.2" | bc)
# to integer
count_test=${count_test%.*}

head -n $count_train fs_all.txt > /gpfs/projects/upf97/data/train_fs_mmap.txt
tail -n $count_test fs_all.txt > /gpfs/projects/upf97/data/test_fs_mmap.txt


echo "Files have been written to '/gpfs/projects/upf97/data/fs_train_mmap.txt' and '/gpfs/projects/upf97/data/fs_test_mmap.txt'."