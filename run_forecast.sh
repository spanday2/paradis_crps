#!/bin/bash
#PBS -N paradis_forecast
#PBS -l select=1:ncpus=24:mpiprocs=12:ngpus=4:mem=100G:vntype=gpu
#PBS -l walltime=12:00:00
#PBS -o paradis_forecast.out
#PBS -e paradis_forecast.err

cd "$PBS_O_WORKDIR" || exit 1

echo "Job ID: $PBS_JOBID"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
source /home/shp000/site8/conda/miniforge3/etc/profile.d/conda.sh
conda activate paradis_new

echo "Python: $(which python)"
python --version

echo "Visible GPUs:"
nvidia-smi

# ------------------------------------------------------------
# Forecast
# ------------------------------------------------------------
# -u gives unbuffered Python output
# 2>&1 combines stdout/stderr from forecast.py
# tee writes it to forecast.log while still showing it in PBS stdout
# PIPESTATUS[0] preserves the Python exit status
# ------------------------------------------------------------

python -u forecast.py \
    --config config/paradis_settings.yaml \
    --checkpoint-path /home/shp000/site7/ensemble/paradis_crps/logs/lightning_logs/version_4/checkpoints/'epoch=0025.ckpt' \
    --output-file results/forecast.zarr \
    --num-devices 4 \
    --forecast-steps 40 \
    --flush-every-n-steps 5 \
    2>&1 | tee forecast.log

status=${PIPESTATUS[0]}

echo "Forecast exit status: $status"
echo "End time: $(date)"

exit $status