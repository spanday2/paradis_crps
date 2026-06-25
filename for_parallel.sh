#PBS -S /bin/bash
#PBS -N ens-inference_SN
#PBS -l select=1:ncpus=48:mpiprocs=4:ngpus=4:mem=437G:vntype=gpu
#PBS -l walltime=12:00:00
#PBS -o forecastStage2.log
#PBS -e forecastStage2_error.log

set -euo pipefail

export MASTER_PORT=8151

# Unique allocated nodes, preserving PBS order.
NODES=$(awk '!seen[$0]++' "$PBS_NODEFILE")

# First allocated node is the master.
MASTER_ADDR=$(awk '!seen[$0]++ {print; exit}' "$PBS_NODEFILE")
export MASTER_ADDR

# Total number of GPU workers.
# With select=2 and mpiprocs=4, this should be 8.
export WORLD_SIZE=$(wc -l < "$PBS_NODEFILE")

export OMP_NUM_THREADS=1

echo "MASTER_ADDR is: $MASTER_ADDR"
echo "MASTER_PORT is: $MASTER_PORT"
echo "WORLD_SIZE is: $WORLD_SIZE"
echo "NODES is:"
echo "$NODES"
echo "OMP_NUM_THREADS is: $OMP_NUM_THREADS"

NODE_RANK=0

for NODE in $NODES; do
    echo "Starting forecast parent on $NODE with NODE_RANK $NODE_RANK"

    ssh "$NODE" "
        set -euo pipefail

        export MASTER_ADDR=\"$MASTER_ADDR\";
        export MASTER_PORT=\"$MASTER_PORT\";
        export WORLD_SIZE=\"$WORLD_SIZE\";
        export NODE_RANK=\"$NODE_RANK\";
        export OMP_NUM_THREADS=\"$OMP_NUM_THREADS\";

        cd /home/shp000/site7/ensemble/paradis_crps/

        echo \"On \$HOSTNAME\"
        echo \"NODE_RANK=\$NODE_RANK\"
        echo \"MASTER_ADDR=\$MASTER_ADDR\"
        echo \"MASTER_PORT=\$MASTER_PORT\"
        echo \"WORLD_SIZE=\$WORLD_SIZE\"
        echo \"OMP_NUM_THREADS=\$OMP_NUM_THREADS\"

        /home/shp000/site8/conda/miniforge3/envs/paradis/bin/python forecast_parallel.py \
            config/paradis_forecast.yaml \
            > forecast_stage2_node${NODE_RANK}.log 2>&1
    " &

    NODE_RANK=$((NODE_RANK + 1))
done

wait

echo "Forecast finished."