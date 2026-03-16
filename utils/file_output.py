import os
import shutil
import dask
import numpy
import xarray

from utils.mhuaes import mhuaes3


def save_results_to_zarr(
    data,
    ds_input_data,
    atmospheric_vars,
    surface_vars,
    constant_vars,
    dataset,
    pressure_levels,
    filename,
    ind,
    init_times,
    ensemble_mode: bool = False,
):
    """Save results to a Zarr file.

    Args:
        data:          In deterministic mode: (B, output_steps, F, lat, lon)
                       In ensemble mode:      (B, M, output_steps, F, lat, lon)
        ensemble_mode: If True, data has an extra member dimension (M) and
                       a 'member' coordinate is added to the output dataset.
        All other args unchanged from the original.
    """
    data_vars = {}
    num_levels = len(pressure_levels)

    input_data = ds_input_data.sel(time=init_times).sortby("time")["data"].values

    if ensemble_mode:
        # data shape: (B, M, output_steps, F, lat, lon)
        num_members = data.shape[1]

        # Atmospheric variables — add member dim
        # input_data has no member dim so we broadcast it across all members
        # final shape per var: (B, M, output_steps+1, levels, lat, lon)
        atm_dims = ["time", "member", "prediction_timedelta", "level", "latitude", "longitude"]
        for i, feature in enumerate(atmospheric_vars):
            beg_ind = i * num_levels
            end_ind = (i + 1) * num_levels

            # input slice: (B, 1, levels, lat, lon) — broadcast across members
            input_slice = (
                input_data[..., beg_ind:end_ind]
                .transpose(0, 3, 1, 2)[:, None, None]         # (B, 1, 1, levels, lat, lon)
                .repeat(num_members, axis=1)                   # (B, M, 1, levels, lat, lon)
            )
            # forecast slice: (B, M, output_steps, levels, lat, lon)
            forecast_slice = data[:, :, :, beg_ind:end_ind]   # (B, M, steps, lat, lon, levels)

            data_vars[feature] = (
                atm_dims,
                numpy.concatenate((input_slice, forecast_slice), axis=2),
            )

        # Surface variables — add member dim
        # final shape per var: (B, M, output_steps+1, lat, lon)
        sur_dims = ["time", "member", "prediction_timedelta", "latitude", "longitude"]
        for i, feature in enumerate(surface_vars):
            if feature == "wind_z_10m":
                continue
            feat_idx = len(atmospheric_vars) * num_levels + i

            # input slice: (B, 1, 1, lat, lon) → broadcast across members
            input_slice = (
                input_data[..., feat_idx][:, None, None]       # (B, 1, 1, lat, lon)
                .repeat(num_members, axis=1)                   # (B, M, 1, lat, lon)
            )
            # forecast slice: (B, M, output_steps, lat, lon)
            forecast_slice = data[:, :, :, feat_idx]

            data_vars[feature] = (
                sur_dims,
                numpy.concatenate((input_slice, forecast_slice), axis=2),
            )

        coords_extra = {"member": numpy.arange(num_members)}

    else:
        # Deterministic — original behaviour unchanged
        # data shape: (B, output_steps, F, lat, lon)

        atm_dims = ["time", "prediction_timedelta", "level", "latitude", "longitude"]
        for i, feature in enumerate(atmospheric_vars):
            beg_ind = i * num_levels
            end_ind = (i + 1) * num_levels

            data_vars[feature] = (
                atm_dims,
                numpy.concatenate(
                    (
                        input_data[..., beg_ind:end_ind].transpose(0, 3, 1, 2)[:, None],
                        data[:, :, beg_ind:end_ind],
                    ),
                    axis=1,
                ),
            )

        sur_dims = ["time", "prediction_timedelta", "latitude", "longitude"]
        for i, feature in enumerate(surface_vars):
            if feature == "wind_z_10m":
                continue
            data_vars[feature] = (
                sur_dims,
                numpy.concatenate(
                    (
                        input_data[..., len(atmospheric_vars) * num_levels + i][:, None],
                        data[:, :, len(atmospheric_vars) * num_levels + i],
                    ),
                    axis=1,
                ),
            )

        coords_extra = {}

    if ind == 0:
        # Constant variables — no member dim, same in both modes
        con_dims = ["latitude", "longitude"]
        for i, feature in enumerate(dataset.ds_constants.data_vars):
            if feature in con_dims:
                continue
            data_vars[feature] = (con_dims, dataset.ds_constants[feature].data)

    # Number of output steps depends on mode
    num_output_steps = data.shape[2] if ensemble_mode else data.shape[1]

    # Define coordinates
    coords = {
        "latitude": dataset.lat,
        "longitude": dataset.lon,
        "time": init_times,
        "level": pressure_levels,
        "prediction_timedelta": (numpy.arange(num_output_steps + 1))
        * numpy.timedelta64(6 * 3600 * 10**9, "ns"),
        **coords_extra,
    }

    # If this is the first write, remove any existing Zarr store
    if ind == 0 and os.path.exists(filename):
        shutil.rmtree(filename)

    # Create dataset
    ds = xarray.Dataset(data_vars=data_vars, coords=coords)

    # Add dewpoint depression to files
    hu = ds.specific_humidity
    tt = ds.temperature
    ps = ds.level * 100
    ds = ds.assign(dewpoint_depression=mhuaes3(hu, tt, ps))

    with dask.config.set(scheduler="threads"):

        if ind == 0:
            encoding = {
                "time": {"dtype": "float64"},
            }

            for var in ds.data_vars:
                if "time" in ds[var].dims:
                    var_shape = ds[var].shape
                    encoding[var] = {
                        "chunks": (
                            1,
                            *var_shape[1:],
                        ),
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