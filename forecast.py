import sys
from datetime import datetime
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import numpy

from trainer import LitParadis
from data.datamodule import Era5DataModule
from utils.file_output import save_results_to_zarr
from utils.postprocessing import (
    denormalize_datasets,
    convert_cartesian_to_spherical_winds,
    replace_variable_name,
)
from utils.visualization import plot_forecast_map


def main():
    """Generate forecasts using a trained model.
    Usage: python forecast.py path/to/config_file.yaml
    """

    cfg = OmegaConf.load(sys.argv[1])

    """
    Core forecast execution logic.

    Args:
        cfg: Fully configured DictConfig with all necessary parameters set
    """
    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.compute.accelerator == "gpu"
        else "cpu"
    )

    # Initialize data module
    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="predict")
    dataset = datamodule.dataset

    # Extract features and dimensions
    atmospheric_vars = cfg.features.output.atmospheric
    surface_vars = cfg.features.output.surface
    constant_vars = cfg.features.input.constants
    pressure_levels = cfg.features.pressure_levels

    num_levels = len(pressure_levels)
    num_atm_features = len(atmospheric_vars) * num_levels
    num_sur_features = len(surface_vars)
    num_features = num_atm_features + num_sur_features
    num_forecast_steps = cfg.model.forecast_steps

    # Ensemble configuration — mirrors LitParadis setup
    noise_channels  = cfg.model.get("noise_channels", 0)
    ensemble_mode   = noise_channels > 0
    num_members     = cfg.training.get("num_ensemble_members", 4) if ensemble_mode else 1

    # Get the output number of forecast steps based on the output frequency
    output_frequency = cfg.forecast.output_frequency
    output_num_forecast_steps = max(1, num_forecast_steps // output_frequency)

    output_features = list(dataset.dyn_output_features)

    # Load model
    litmodel = LitParadis(datamodule, cfg)
    if not cfg.init.checkpoint_path:
        raise ValueError(
            "checkpoint_path must be specified in the config for forecasting"
        )

    litmodel.to(device).eval()

    # Rename variables that require post-processing in dataset
    atmospheric_vars = replace_variable_name(
        "wind_x", "u_component_of_wind", atmospheric_vars
    )
    atmospheric_vars = replace_variable_name(
        "wind_y", "v_component_of_wind", atmospheric_vars
    )
    atmospheric_vars = replace_variable_name(
        "wind_z", "vertical_velocity", atmospheric_vars
    )

    surface_vars = replace_variable_name(
        "wind_x_10m", "10m_u_component_of_wind", surface_vars
    )
    surface_vars = replace_variable_name(
        "wind_y_10m", "10m_v_component_of_wind", surface_vars
    )

    # Compute initialization times from dataset
    init_times = dataset.time

    logging.info(f"Number of forecasts to generate: {len(init_times)}")
    if ensemble_mode:
        logging.info(f"Ensemble mode: {num_members} members")

    # Run forecast
    logging.info("Generating forecast...")
    ind = 0
    with torch.inference_mode(), torch.no_grad():
        time_start_ind = 0
        for input_data, ground_truth in tqdm(
            datamodule.predict_dataloader()
        ):

            batch_size = input_data.shape[0]

            if ensemble_mode:
                output_forecast = torch.empty(
                    (
                        batch_size,
                        num_members,
                        output_num_forecast_steps,
                        num_features,
                        dataset.lat_size,
                        dataset.lon_size,
                    ),
                    device=device,
                )

                # Each member maintains its own independent autoregressive state
                member_inputs = [input_data.clone() for _ in range(num_members)]

                for m in range(num_members):
                    frequency_counter = 0
                    for step in range(num_forecast_steps):
                        # model.forward() auto-samples per-grid-point noise when
                        # noise_emb is not provided — each member call gets fresh noise
                        output_data = litmodel(
                            member_inputs[m][:, step].to(device),
                        )
                        
                        # zero_noise_emb = torch.zeros(
                        #     batch_size,
                        #     litmodel.model.hidden_dim,
                        #     litmodel.nlat,
                        #     litmodel.nlon,
                        #     device=device,
                        #     dtype=member_inputs[m].dtype,
                        # )
                        
                        # output_data = litmodel(
                        #     member_inputs[m][:, step].to(device),
                        #     noise_emb=zero_noise_emb, 
                        # )
                        
                        member_inputs[m] = litmodel._autoregression_input_from_output(
                            member_inputs[m], output_data, step, num_forecast_steps
                        )

                        if step % cfg.forecast.output_frequency == 0:
                            output_forecast[:, m, frequency_counter] = output_data
                            frequency_counter += 1

                # Transfer to CPU: (B, M, output_steps, F, lat, lon)
                # output_forecast = output_forecast.cpu()
                output_forecast = output_forecast.cpu()

                # Denormalize each member independently
                for m in range(num_members):
                    denormalize_datasets(ground_truth, output_forecast[:, m], dataset)

                output_forecast = output_forecast.numpy().astype(numpy.float64)
                ground_truth_np = ground_truth.numpy().astype(numpy.float64)
                # Post-process winds for each member
                for m in range(num_members):
                    convert_cartesian_to_spherical_winds(
                        dataset.lat, dataset.lon, cfg, ground_truth_np, output_features
                    )
                    # convert_cartesian_to_spherical_winds(
                    #     dataset.lat, dataset.lon, cfg, output_forecast[:, m].numpy(), output_features
                    # )
                    convert_cartesian_to_spherical_winds(
                        dataset.lat, dataset.lon, cfg,
                        numpy.ascontiguousarray(output_forecast[:, m]),
                        output_features
                    )

                # output_forecast = output_forecast.numpy()

            else:
                # Deterministic forecast
                output_forecast = torch.empty(
                    (
                        batch_size,
                        output_num_forecast_steps,
                        num_features,
                        dataset.lat_size,
                        dataset.lon_size,
                    ),
                    device=device,
                )

                frequency_counter = 0
                for step in range(num_forecast_steps):
                    output_data = litmodel(
                        input_data[:, step].to(device),
                    )

                    input_data = litmodel._autoregression_input_from_output(
                        input_data, output_data, step, num_forecast_steps
                    )

                    if step % cfg.forecast.output_frequency == 0:
                        output_forecast[:, frequency_counter] = output_data
                        frequency_counter += 1

                output_forecast = output_forecast.cpu()
                denormalize_datasets(ground_truth, output_forecast, dataset)
                output_forecast = output_forecast.numpy()

                convert_cartesian_to_spherical_winds(
                    dataset.lat, dataset.lon, cfg, ground_truth, output_features
                )
                convert_cartesian_to_spherical_winds(
                    dataset.lat, dataset.lon, cfg, output_forecast, output_features
                )

            # Save results
            if cfg.forecast.output_file is not None:
                save_results_to_zarr(
                    output_forecast,
                    dataset.ds_loader,
                    atmospheric_vars,
                    surface_vars,
                    constant_vars,
                    dataset,
                    pressure_levels,
                    cfg.forecast.output_file,
                    ind,
                    init_times[time_start_ind : time_start_ind + batch_size],
                    ensemble_mode=ensemble_mode,
                )

            ind += 1
            time_start_ind += batch_size

    logging.info("Saved output files successfuly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()