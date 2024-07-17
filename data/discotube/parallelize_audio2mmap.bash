

# for file in discotube-2020-09
for subdir in /gpfs/projects/upf97/discotube/discotube-2020-09/audio-new/audio/
do
    echo $file
    sbatch launch_mmap.bash $subdir 0
done


# for file in discotube-2023-03
for subdir in /gpfs/projects/upf97/discotube/discotube-2023-03/audio-new/audio/
do
    echo $file
    sbatch launch_mmap.bash $subdir 1
done