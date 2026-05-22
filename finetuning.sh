#PBS -S /bin/bash
#PBS -N ens_1deg_finetune
#PBS -l select=4:ncpus=48:mpiprocs=4:ngpus=4:mem=437G:vntype=gpu
#PBS -l walltime=12:00:00
#PBS -o ens_1deg_finetune.log
#PBS -e ens_1deg_finetune_error.log

set -euo pipefail

export MASTER_PORT=8148

NODES=$(sort -u "$PBS_NODEFILE")
MASTER_ADDR=$(echo "$NODES" | head -n 1)
export WORLD_SIZE=$(wc -l < "$PBS_NODEFILE")
export OMP_NUM_THREADS=1

echo "MASTER_ADDR is: $MASTER_ADDR"
echo "MASTER_PORT is: $MASTER_PORT"
echo "WORLD_SIZE is: $WORLD_SIZE"
echo "NODES is: $NODES"
echo "OMP_NUM_THREADS is: $OMP_NUM_THREADS"

PY=/home/shp000/site8/conda/miniforge3/envs/paradis/bin/python
CODEDIR=/home/shp000/site7/ensemble/paradis_crps

FIRST_CKPT=/home/shp000/site7/ensemble/paradis_crps/logs/lightning_logs/version_82/checkpoints/0003500.ckpt

PREV_CKPT="$FIRST_CKPT"

for FS in 8 9 10 11 12; do

    EXP="ens_1deg_stage_${FS}"

    echo "======================================================"
    echo "Starting forecast_steps=${FS}"
    echo "Experiment name: ${EXP}"
    echo "Initial checkpoint: ${PREV_CKPT}"
    echo "======================================================"

    if [ ! -f "$PREV_CKPT" ]; then
        echo "ERROR: checkpoint not found: $PREV_CKPT"
        exit 1
    fi

    STAGE_START_TIME=$(date +%s)

    NODE_RANK=0

    for NODE in $NODES; do
        echo "Starting training on $NODE with NODE_RANK $NODE_RANK"

        ssh -n -o StrictHostKeyChecking=no "$NODE" "
            set -euo pipefail

            export MASTER_ADDR=\"$MASTER_ADDR\"
            export MASTER_PORT=\"$MASTER_PORT\"
            export WORLD_SIZE=\"$WORLD_SIZE\"
            export NODE_RANK=\"$NODE_RANK\"
            export OMP_NUM_THREADS=\"$OMP_NUM_THREADS\"

            cd \"$CODEDIR\"

            mkdir -p run_outputs

            echo \"On \$HOSTNAME with NODE_RANK=\$NODE_RANK MASTER_ADDR=\$MASTER_ADDR MASTER_PORT=\$WORLD_SIZE\"

            $PY train.py \
                model.forecast_steps=$FS \
                compute.num_nodes=4 \
                compute.num_devices=4 \
                training.max_steps=3500 \
                init.checkpoint_path=\"\\\"$PREV_CKPT\\\"\" \
                init.restart=false \
                > run_outputs/output_${EXP}_rank${NODE_RANK}.log 2>&1
        " &

        NODE_RANK=$((NODE_RANK + 1))
    done

    if ! wait; then
        echo "ERROR: one or more node processes failed for forecast_steps=${FS}"
        echo "The script will stop here and will NOT continue to the next FS."
        echo "Check these logs:"
        echo "$CODEDIR/run_outputs/output_${EXP}_rank0.log"
        echo "$CODEDIR/run_outputs/output_${EXP}_rank1.log"
        echo "$CODEDIR/run_outputs/output_${EXP}_rank2.log"
        echo "$CODEDIR/run_outputs/output_${EXP}_rank3.log"
        exit 1
    fi

    echo "Finished forecast_steps=${FS}"

    NEW_CKPT=$(find "$CODEDIR/logs/lightning_logs" \
        -path "*/checkpoints/*.ckpt" \
        -type f \
        -newermt "@${STAGE_START_TIME}" \
        -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$NEW_CKPT" ]; then
        echo "ERROR: no new checkpoint was created for forecast_steps=${FS}"
        echo "The script will stop here and will NOT continue to the next FS."
        echo "Check:"
        echo "$CODEDIR/run_outputs/output_${EXP}_rank0.log"
        exit 1
    fi

    PREV_CKPT="$NEW_CKPT"

    echo "Next stage will initialize from:"
    echo "$PREV_CKPT"

done

echo "All fine-tuning stages finished."