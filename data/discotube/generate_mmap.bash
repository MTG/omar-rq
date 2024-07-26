#!/bin/bash

# Set the directory
directory="/gpfs/scratch/upf97/discotube"
find "$directory" -type f -name "*.mmap" > all.txt
count=$(wc -l < all.txt)
echo "Total number of files: $count"

count_train=$(echo "scale=0; $count * 0.8" | bc)
# to integer
count_train=${count_train%.*}
count_test=$(echo "scale=0; $count * 0.2" | bc)
# to integer
count_test=${count_test%.*}

head -n $count_train all.txt > /gpfs/projects/upf97/data/train_mmap.txt
tail -n $count_test all.txt > /gpfs/projects/upf97/data/test_mmap.txt


echo "Files have been written to '/gpfs/projects/upf97/data/train_mmap.txt' and '/gpfs/projects/upf97/data/test_mmap.txt'."