#!/bin/bash

# Set the directory
directory="segments"
find "$directory" -type f -name "*.txt" > all.txt

# Remove .mp3 extension from each path
sed -i 's/.txt//g' all.txt

# Remove path only keep basename
sed -i 's/.*\///' all.txt

# Get total number of files
count=$(wc -l < all.txt)
echo "Total number of files: $count"

# Calculate the number of files for training, validation, and testing
count_train=$(echo "scale=0; $count * 0.6" | bc)
count_train=${count_train%.*}

count_validation=$(echo "scale=0; $count * 0.2" | bc)
count_validation=${count_validation%.*}

count_test=$(echo "scale=0; $count * 0.2" | bc)
count_test=${count_test%.*}

# Create the training set
head -n $count_train all.txt > train.txt

# Create the validation set
tail -n +$(($count_train + 1)) all.txt | head -n $count_validation > validation.txt

# Create the test set
tail -n $count_test all.txt > test.txt

echo "Files have been written to 'train.txt', 'validation.txt', and 'test.txt'."
