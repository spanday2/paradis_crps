import os
import sys
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from trainer import LitParadis
from data.datamodule import Era5DataModule
from utils.file_output import save_results_to_zarr
from utils.postprocessing import (
    denormalize_datasets,
    convert_cartesian_to_spherical_winds,
    replace_variable_name,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_forecast_seed(seed: int) -> None:
    """Set random seeds for reproducible forecasting."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------
def is_distributed_env() -> bool:
    """Return True if PBS script has set training-style distributed variables."""
    required = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "NODE_RANK"]
    return all(k in os.environ for k in required)


def make_rank_output_file(output_file: str, rank: int, world_size: int) -> str:
    """Create one Zarr output path per global rank and ensure directory exists."""
    path = Path(output_file)

    if path.suffix == ".zarr":
        rank_path = path.with_name(
            f"{path.stem}_rank{rank:03d}_of_{world_size:03d}.zarr"
        )
    else:
        rank_path = Path(f"{output_file}_rank{rank:03d}_of_{world_size:03d}.zarr")

    rank_path.parent.mkdir(parents=True, exist_ok=True)

    return str(rank_path)


def write_member_map(
    output_file: str,
    rank: int,
    world_size: int,
    members: list[int],
) -> None:
    """Save which global ensemble members are contained in this rank's Zarr."""
    path = Path(output_file)

    if path.suffix == ".zarr":
        map_path = path.with_name(
            f"{path.stem}_rank{rank:03d}_of_{world_size:03d}_members.json"
        )
    else:
        map_path = Path(
            f"{output_file}_rank{rank:03d}_of_{world_size:03d}_members.json"
        )

    map_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "rank": rank,
        "world_size": world_size,
        "members": members,
    }

    with open(map_path, "w") as f:
        json.dump(payload, f, indent=2)


def setup_worker_distributed(
    local_rank: int,
    cfg: DictConfig,
) -> tuple[bool, int, int, torch.device]:
    """Initialize one local GPU worker.

    PBS launches one parent forecast.py per node.
    Each parent spawns one local worker per GPU.

    Global rank is computed as:
        global_rank = NODE_RANK * compute.num_devices + local_rank
    """

    num_devices = int(cfg.compute.get("num_devices", 1))

    distributed = (
        is_distributed_env()
        and int(os.environ["WORLD_SIZE"]) > 1
    )

    if distributed:
        node_rank = int(os.environ["NODE_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = node_rank * num_devices + local_rank

        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend="nccl",
            init_method=(
                f"tcp://{os.environ['MASTER_ADDR']}:"
                f"{os.environ['MASTER_PORT']}"
            ),
            rank=global_rank,
            world_size=world_size,
        )

        return True, global_rank, world_size, device

    rank = 0
    world_size = 1

    if torch.cuda.is_available() and cfg.compute.accelerator == "gpu":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return False, rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------
# Forecast worker
# ---------------------------------------------------------------------
def forecast_worker(local_rank: int, cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)

    distributed, rank, world_size, device = setup_worker_distributed(
        local_rank,
        cfg,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {rank}] %(asctime)s %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------
    base_seed = int(cfg.init.get("seed", 43))
    set_forecast_seed(base_seed + rank)

    logging.info(f"Forecast base seed = {base_seed}")
    logging.info(f"Distributed = {distributed}")
    logging.info(f"Rank = {rank}")
    logging.info(f"World size = {world_size}")
    logging.info(f"Local rank = {local_rank}")
    logging.info(f"Device = {device}")

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="predict")
    dataset = datamodule.dataset

    atmospheric_vars = cfg.features.output.atmospheric
    surface_vars = cfg.features.output.surface
    constant_vars = cfg.features.input.constants
    pressure_levels = cfg.features.pressure_levels

    num_levels = len(pressure_levels)
    num_atm_features = len(atmospheric_vars) * num_levels
    num_sur_features = len(surface_vars)
    num_features = num_atm_features + num_sur_features

    num_forecast_steps = int(cfg.model.forecast_steps)
    output_frequency = int(cfg.forecast.output_frequency)

    # More robust than num_forecast_steps // output_frequency.
    output_num_forecast_steps = len(
        range(0, num_forecast_steps, output_frequency)
    )

    output_features = list(dataset.dyn_output_features)

    # ------------------------------------------------------------
    # Ensemble setup
    # ------------------------------------------------------------
    noise_channels = cfg.model.get("noise_channels", 0)
    ensemble_mode = noise_channels > 0

    num_members = (
        int(cfg.training.get("num_ensemble_members", 1))
        if ensemble_mode
        else 1
    )

    if ensemble_mode:
        all_members = list(range(num_members))
        local_members = all_members[rank::world_size]
    else:
        # Deterministic mode: only rank 0 works.
        local_members = [0] if rank == 0 else []

    logging.info(f"Total ensemble members = {num_members}")
    logging.info(f"Local members for this rank = {local_members}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    litmodel = LitParadis(datamodule, cfg)

    if not cfg.init.checkpoint_path:
        raise ValueError(
            "cfg.init.checkpoint_path must be specified for forecasting."
        )

    litmodel.to(device).eval()

    # ------------------------------------------------------------
    # Rename variables for output
    # ------------------------------------------------------------
    atmospheric_vars = replace_variable_name(
        "wind_x",
        "u_component_of_wind",
        atmospheric_vars,
    )
    atmospheric_vars = replace_variable_name(
        "wind_y",
        "v_component_of_wind",
        atmospheric_vars,
    )
    atmospheric_vars = replace_variable_name(
        "wind_z",
        "vertical_velocity",
        atmospheric_vars,
    )

    surface_vars = replace_variable_name(
        "wind_x_10m",
        "10m_u_component_of_wind",
        surface_vars,
    )
    surface_vars = replace_variable_name(
        "wind_y_10m",
        "10m_v_component_of_wind",
        surface_vars,
    )

    init_times = dataset.time

    if cfg.forecast.output_file is not None:
        rank_output_file = make_rank_output_file(
            cfg.forecast.output_file,
            rank,
            world_size,
        )

        logging.info(f"This rank will write: {rank_output_file}")

        write_member_map(
            cfg.forecast.output_file,
            rank,
            world_size,
            local_members,
        )
    else:
        rank_output_file = None

    # ------------------------------------------------------------
    # Forecast loop
    # ------------------------------------------------------------
    logging.info("Generating forecast...")

    with torch.inference_mode(), torch.no_grad():
        time_start_ind = 0
        ind = 0

        dataloader = datamodule.predict_dataloader()
        dataloader = tqdm(
            dataloader,
            desc=f"rank {rank}",
            position=rank,
            disable=False,
        )

        for input_data, ground_truth in dataloader:
            batch_size = input_data.shape[0]

            # ====================================================
            # Ensemble forecast
            # ====================================================
            if ensemble_mode:
                if len(local_members) > 0:
                    output_forecast = torch.empty(
                        (
                            batch_size,
                            len(local_members),
                            output_num_forecast_steps,
                            num_features,
                            dataset.lat_size,
                            dataset.lon_size,
                        ),
                        device=device,
                    )

                    # Each local member keeps its own autoregressive state.
                    member_inputs = [input_data.clone() for _ in local_members]

                    for local_m, global_m in enumerate(local_members):
                        # Stable member-specific seed.
                        # Member identity remains stable independent of world_size.
                        member_seed = base_seed + 1000003 * global_m + ind
                        torch.manual_seed(member_seed)

                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(member_seed)

                        frequency_counter = 0

                        for step in range(num_forecast_steps):
                            output_data = litmodel(
                                member_inputs[local_m][:, step].to(device)
                            )

                            member_inputs[local_m] = (
                                litmodel._autoregression_input_from_output(
                                    member_inputs[local_m],
                                    output_data,
                                    step,
                                    num_forecast_steps,
                                )
                            )

                            if step % output_frequency == 0:
                                output_forecast[
                                    :, local_m, frequency_counter
                                ] = output_data
                                frequency_counter += 1

                    output_forecast = output_forecast.cpu()

                    # Denormalize each local member independently.
                    for local_m in range(len(local_members)):
                        denormalize_datasets(
                            ground_truth,
                            output_forecast[:, local_m],
                            dataset,
                        )

                    output_forecast = output_forecast.numpy().astype(np.float64)
                    ground_truth_np = ground_truth.numpy().astype(np.float64)

                    # Convert truth once for this rank.
                    convert_cartesian_to_spherical_winds(
                        dataset.lat,
                        dataset.lon,
                        cfg,
                        ground_truth_np,
                        output_features,
                    )

                    # Convert each local ensemble member.
                    for local_m in range(len(local_members)):
                        convert_cartesian_to_spherical_winds(
                            dataset.lat,
                            dataset.lon,
                            cfg,
                            np.ascontiguousarray(output_forecast[:, local_m]),
                            output_features,
                        )

                    if rank_output_file is not None:
                        save_results_to_zarr(
                            output_forecast,
                            dataset.ds_loader,
                            atmospheric_vars,
                            surface_vars,
                            constant_vars,
                            dataset,
                            pressure_levels,
                            rank_output_file,
                            ind,
                            init_times[
                                time_start_ind : time_start_ind + batch_size
                            ],
                            ensemble_mode=True,
                        )

            # ====================================================
            # Deterministic forecast
            # ====================================================
            else:
                if rank == 0:
                    input_data = input_data.to(device)

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
                        output_data = litmodel(input_data[:, step])

                        input_data = litmodel._autoregression_input_from_output(
                            input_data,
                            output_data,
                            step,
                            num_forecast_steps,
                        )

                        if step % output_frequency == 0:
                            output_forecast[:, frequency_counter] = output_data
                            frequency_counter += 1

                    output_forecast = output_forecast.cpu()

                    denormalize_datasets(
                        ground_truth,
                        output_forecast,
                        dataset,
                    )

                    output_forecast = output_forecast.numpy().astype(np.float64)
                    ground_truth_np = ground_truth.numpy().astype(np.float64)

                    convert_cartesian_to_spherical_winds(
                        dataset.lat,
                        dataset.lon,
                        cfg,
                        ground_truth_np,
                        output_features,
                    )

                    convert_cartesian_to_spherical_winds(
                        dataset.lat,
                        dataset.lon,
                        cfg,
                        output_forecast,
                        output_features,
                    )

                    if rank_output_file is not None:
                        save_results_to_zarr(
                            output_forecast,
                            dataset.ds_loader,
                            atmospheric_vars,
                            surface_vars,
                            constant_vars,
                            dataset,
                            pressure_levels,
                            rank_output_file,
                            ind,
                            init_times[
                                time_start_ind : time_start_ind + batch_size
                            ],
                            ensemble_mode=False,
                        )

            if distributed:
                dist.barrier()

            ind += 1
            time_start_ind += batch_size

    cleanup_distributed(distributed)

    logging.info("Forecast finished successfully.")


# ---------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python forecast.py config/paradis_forecast.yaml"
        )

    cfg_path = sys.argv[1]
    cfg = OmegaConf.load(cfg_path)

    num_devices = int(cfg.compute.get("num_devices", 1))

    # ------------------------------------------------------------
    # PBS/training-style distributed launch:
    # one parent forecast.py per node, each parent spawns local GPUs.
    #
    # Required environment variables:
    #   MASTER_ADDR
    #   MASTER_PORT
    #   WORLD_SIZE
    #   NODE_RANK
    # ------------------------------------------------------------
    if is_distributed_env() and int(os.environ["WORLD_SIZE"]) > 1:
        if torch.cuda.is_available() and cfg.compute.accelerator == "gpu":
            available_gpus = torch.cuda.device_count()

            if num_devices > available_gpus:
                raise RuntimeError(
                    f"Requested compute.num_devices={num_devices}, "
                    f"but only {available_gpus} CUDA devices are visible."
                )

        mp.spawn(
            forecast_worker,
            args=(cfg_path,),
            nprocs=num_devices,
            join=True,
        )
    else:
        forecast_worker(0, cfg_path)


if __name__ == "__main__":
    main()