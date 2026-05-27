#PBS -S /bin/bash
#PBS -N ens_1deg_stage_1
#PBS -l select=4:ncpus=48:mpiprocs=4:ngpus=4:mem=437G:vntype=gpu
#PBS -l walltime=12:00:00
#PBS -o ens_1deg.log
#PBS -e ens_1deg_error.log

export MASTER_PORT=8148
NODES=$(cat $PBS_NODEFILE | sort | uniq)
MASTER_ADDR=$(echo $NODES | cut -d ' ' -f 1)
export WORLD_SIZE=$(wc -l < $PBS_NODEFILE)
export OMP_NUM_THREADS=1


echo "  MASTER_ADDR is: $MASTER_ADDR"
echo "  MASTER_PORT is: $MASTER_PORT"
echo "  WORLD_SIZE is: $WORLD_SIZE"
echo "  NODES is: $NODES"
echo "  OMP_NUM_THREADS is: $OMP_NUM_THREADS"

NODE_RANK=0
for NODE in $NODES; do
    echo "Starting training on $NODE with NODE_RANK $NODE_RANK, MASTER_ADDR $MASTER_ADDR, MASTER_PORT $MASTER_PORT, OMP_NUM_THREADS $OMP_NUM_THREADS"
    ssh $NODE "
        set -euo pipefail
        export MASTER_ADDR=\"$MASTER_ADDR\";
        export MASTER_PORT=\"$MASTER_PORT\";
        export WORLD_SIZE=\"$WORLD_SIZE\";
        export NODE_RANK=\"$NODE_RANK\";
        export OMP_NUM_THREADS=\"$OMP_NUM_THREADS\";

        cd /home/shp000/site7/ensemble/paradis_crps/

        echo \"   On \$HOSTNAME with NODE_RANK \$NODE_RANK, MASTER_ADDR \$MASTER_ADDR, MASTER_PORT \$MASTER_PORT, WORLD_SIZE \$WORLD_SIZE, OMP_NUM_THREADS \$OMP_NUM_THREADS\";
        /home/shp000/site8/conda/miniforge3/envs/paradis/bin/python train.py > output_1deg.log \
            compute.num_nodes=4 \
            compute.num_devices=4 \
    " &
    NODE_RANK=$((NODE_RANK + 1))
done

wait

echo "All remote processes finished."
