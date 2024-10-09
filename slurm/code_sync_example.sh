grep -v '^#' .gitignore > rsync-exclude
rsync -avz --exclude-from=rsync-exclude . alogin1:~/ssl-mtg
rm rsync-excludecd