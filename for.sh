#PBS -S /bin/bash
#PBS -N ens-inference
#PBS -l select=1:ncpus=48:mpiprocs=4:ngpus=4:mem=437G:vntype=gpu
#PBS -l walltime=12:00:00
#PBS -o forecast_43.log
#PBS -e forecast_43_error.log

export OMP_NUM_THREADS=1

echo "Starting forecast on $(hostname)"
echo "OMP_NUM_THREADS is: $OMP_NUM_THREADS"

cd /home/shp000/site7/ensemble/paradis_crps/

/home/shp000/site8/conda/miniforge3/envs/paradis/bin/python forecast.py \
    config/paradis_forecast.yaml \
    > forecast_output_43.log 2>&1

echo "Forecast finished."