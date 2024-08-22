# Download logs from a server and sync wandb logs 
#
# Usage:
# ./wandb_sync_example.sh
#
# additionally, run the script each -n seconds to keep the logs updated
# watch -n 60 ./wandb_sync_example.sh

set -e

server_name=alogin1
max_minutes=120
folder=/gpfs/projects/upf97/logs/wandb/

mkdir -p wandb/
 
# keep a synced copy of all logs since rsync does not support filtering options
rsync -avz $server_name:$folder wandb_all

# find logs that were updated less than max_minutes ago
keep=$(find wandb_all/ -type f -mmin -$max_minutes -exec dirname {} \; | sed 's|/logs$||g' | sed 's|/files$||g' | sort | uniq)

# remove wandb from kept folders
keep=$(echo $keep | sed 's|wandb\ ||g')
echo "Keeping: $keep"

# copy young logs to the wandb folder 
for d in $keep; do cp -r $d wandb/; done
cp -r wandb_all/debug* wandb/
cp -r wandb_all/latest-run wandb/

# sync logs on wandb/
wandb sync --sync-all wandb/

