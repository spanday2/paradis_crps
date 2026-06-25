import os
import shutil

import dask
import numpy
import xarray

from utils.mhuaes import mhuaes3


def save_results_to_zarr(
    data,
    atmospheric_vars,
    surface_vars,
    dataset,
    pressure_levels,
    filename,
    ind,
    init_times,
    ensemble_mode: bool = False,
):
    """Save forecast results to a Zarr file.

    Important convention in this version:
        We do NOT save the initial input/analysis field.

        Therefore:
            prediction_timedelta index 0 = first model forecast output

        If the model forecast step is 6 hours, then physically:
            prediction_timedelta = 0h  -> valid at init + 6h
            prediction_timedelta = 6h  -> valid at init + 12h
            prediction_timedelta = 12h -> valid at init + 18h

    Args:
        data:
            Deterministic mode:
                shape = (B, output_steps, F, lat, lon)

            Ensemble mode:
                shape = (B, M, output_steps, F, lat, lon)

        ds_input_data:
            Kept in the function signature for compatibility, but no longer
            used to prepend the initial input field.

        atmospheric_vars:
            Output atmospheric variable names, for example:
                geopotential
                u_component_of_wind
                v_component_of_wind
                vertical_velocity
                specific_humidity
                temperature

        surface_vars:
            Output surface variable names, for example:
                10m_u_component_of_wind
                10m_v_component_of_wind
                2m_temperature
                mean_sea_level_pressure

        constant_vars:
            Kept for compatibility.

        dataset:
            Dataset object containing latitude, longitude, constants, etc.

        pressure_levels:
            List of pressure levels.

        filename:
            Output Zarr path.

        ind:
            Forecast batch index. If ind == 0, create a new Zarr.
            Otherwise append along time.

        init_times:
            Initialization times for this batch.

        ensemble_mode:
            If True, data has member dimension.
    """

    data_vars = {}
    num_levels = len(pressure_levels)

    # ============================================================
    # Ensemble forecast output
    # ============================================================
    if ensemble_mode:
        # data shape:
        #   (B, M, output_steps, F, lat, lon)
        num_members = data.shape[1]

        atm_dims = [
            "time",
            "member",
            "prediction_timedelta",
            "level",
            "latitude",
            "longitude",
        ]

        for i, feature in enumerate(atmospheric_vars):
            beg_ind = i * num_levels
            end_ind = (i + 1) * num_levels

            # Forecast slice shape:
            #   (B, M, output_steps, level, lat, lon)
            #
            # We do NOT concatenate the initial input anymore.
            forecast_slice = data[:, :, :, beg_ind:end_ind]

            data_vars[feature] = (
                atm_dims,
                forecast_slice,
            )

        sur_dims = [
            "time",
            "member",
            "prediction_timedelta",
            "latitude",
            "longitude",
        ]

        for i, feature in enumerate(surface_vars):
            # Do not save 10m vertical wind if it exists in feature list.
            if feature == "wind_z_10m":
                continue

            feat_idx = len(atmospheric_vars) * num_levels + i

            # Forecast slice shape:
            #   (B, M, output_steps, lat, lon)
            #
            # We do NOT concatenate the initial input anymore.
            forecast_slice = data[:, :, :, feat_idx]

            data_vars[feature] = (
                sur_dims,
                forecast_slice,
            )

        coords_extra = {
            "member": numpy.arange(num_members),
        }

    # ============================================================
    # Deterministic forecast output
    # ============================================================
    else:
        # data shape:
        #   (B, output_steps, F, lat, lon)

        atm_dims = [
            "time",
            "prediction_timedelta",
            "level",
            "latitude",
            "longitude",
        ]

        for i, feature in enumerate(atmospheric_vars):
            beg_ind = i * num_levels
            end_ind = (i + 1) * num_levels

            # Forecast slice shape:
            #   (B, output_steps, level, lat, lon)
            #
            # We do NOT concatenate the initial input anymore.
            forecast_slice = data[:, :, beg_ind:end_ind]

            data_vars[feature] = (
                atm_dims,
                forecast_slice,
            )

        sur_dims = [
            "time",
            "prediction_timedelta",
            "latitude",
            "longitude",
        ]

        for i, feature in enumerate(surface_vars):
            # Do not save 10m vertical wind if it exists in feature list.
            if feature == "wind_z_10m":
                continue

            feat_idx = len(atmospheric_vars) * num_levels + i

            # Forecast slice shape:
            #   (B, output_steps, lat, lon)
            #
            # We do NOT concatenate the initial input anymore.
            forecast_slice = data[:, :, feat_idx]

            data_vars[feature] = (
                sur_dims,
                forecast_slice,
            )

        coords_extra = {}

    # ============================================================
    # Constant variables
    # ============================================================
    if ind == 0:
        con_dims = ["latitude", "longitude"]

        for feature in dataset.ds_constants.data_vars:
            if feature in con_dims:
                continue

            data_vars[feature] = (
                con_dims,
                dataset.ds_constants[feature].data,
            )

    # ============================================================
    # Number of forecast output steps
    # ============================================================
    if ensemble_mode:
        num_output_steps = data.shape[2]
    else:
        num_output_steps = data.shape[1]

    # ============================================================
    # Coordinates
    #
    # Important:
    #   We keep prediction_timedelta starting at 0.
    #
    #   prediction_timedelta = 0h means first saved model forecast.
    #   In your setup, that first model forecast is physically valid +6h.
    # ============================================================
    coords = {
        "latitude": dataset.lat,
        "longitude": dataset.lon,
        "time": init_times,
        "level": pressure_levels,
        "prediction_timedelta": (
            numpy.arange(num_output_steps)
            * numpy.timedelta64(6 * 3600 * 10**9, "ns")
        ),
        **coords_extra,
    }

    # ============================================================
    # Remove old output on first write
    # ============================================================
    if ind == 0 and os.path.exists(filename):
        shutil.rmtree(filename)

    # ============================================================
    # Create dataset
    # ============================================================
    ds = xarray.Dataset(data_vars=data_vars, coords=coords)

    # ============================================================
    # Add dewpoint depression
    # ============================================================
    if "specific_humidity" in ds and "temperature" in ds:
        hu = ds.specific_humidity
        tt = ds.temperature
        ps = ds.level * 100

        ds = ds.assign(
            dewpoint_depression=mhuaes3(hu, tt, ps)
        )

    # ============================================================
    # Chunking
    # ============================================================
    def get_zarr_chunks(da):
        chunks = []

        for dim in da.dims:
            if dim == "time":
                chunks.append(1)

            elif dim == "member":
                chunks.append(1)

            elif dim == "prediction_timedelta":
                chunks.append(1)

            elif dim == "level":
                chunks.append(da.sizes[dim])

            elif dim == "latitude":
                chunks.append(da.sizes[dim])

            elif dim == "longitude":
                chunks.append(da.sizes[dim])

            else:
                chunks.append(da.sizes[dim])

        return tuple(chunks)

    # ============================================================
    # Write Zarr
    # ============================================================
    with dask.config.set(scheduler="threads"):
        if ind == 0:
            encoding = {
                "time": {"dtype": "float64"},
            }

            for var in ds.data_vars:
                encoding[var] = {
                    "chunks": get_zarr_chunks(ds[var]),
                }

            ds.to_zarr(
                filename,
                consolidated=True,
                zarr_format=2,
                encoding=encoding,
            )

        else:
            ds.to_zarr(
                filename,
                consolidated=True,
                append_dim="time",
                zarr_format=2,
            )