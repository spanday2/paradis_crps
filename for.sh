#PBS -S /bin/bash
#PBS -N ens-inference
#PBS -l select=1:ncpus=24:mpiprocs=2:ngpus=2:mem=180G:vntype=gpu
#PBS -l walltime=12:00:00
#PBS -o train_grf_res_3_forecast_grf_res_3.log
#PBS -e train_grf_res_3_forecast_grf_res_3_error.log

export OMP_NUM_THREADS=1

echo "Starting forecast on $(hostname)"
echo "OMP_NUM_THREADS is: $OMP_NUM_THREADS"

cd /home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/

/home/shp000/site8/conda/miniforge3/envs/paradis/bin/python forecast.py \
    config/paradis_forecast.yaml \
    model.noise_type=grf_noise \
    init.checkpoint_path="/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/logs/lightning_logs/version_8/checkpoints/0010000.ckpt" \
    forecast.output_file=./results/train_grf_res_3_forecast_grf_res_3.zarr \
    model.grf_effective_resolution=3 \
    > ./forecast_log/forecast_output_train_grf_res_3_forecast_grf_res_3.log 2>&1

echo "Forecast finished."