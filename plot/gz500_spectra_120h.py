import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch_harmonics as th
import xarray as xr


plt.rcParams.update({"font.size": 16})

checkpoint_zarrs = [
    {
        "label": "GRF noise RES=5",
        "color": "indigo",
        "path": "/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/results/train_grf_noise_forecast_grf_10k_ckpt.zarr",
    },
    {
        "label": "Gaussian noise",
        "color": "darkorange",
        "path": "/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/results/train_raw_noise_forecast_raw_10k_ckpt.zarr",
    },
    {
        "label": "GRF noise RES=1",
        "color": "teal",
        "path": "/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/results/train_grf_res_1_forecast_grf_res_1.zarr",
    },
    {
        "label": "GRF noise RES=3",
        "color": "crimson",
        "path": "/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/results/train_grf_res_3_forecast_grf_res_3.zarr",
    },
]

plot_configs = [
    {
        "var": "geopotential",
        "level": 500,
        "label": "Z500",
    },
    {
        "var": "2m_temperature",
        "level": 0,
        "label": "T2M",
    },
    {
        "var": "specific_humidity",
        "level": 700,
        "label": "Q700",
    },
]

lead_hour_to_plot = 120
truth_lead_hour = 0
max_inits = None
batch_size = 4
convert_geopotential_to_height = False
convert_specific_humidity_to_gkg = True
G0 = 9.80665

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_duplicate_time_index(obj, name="dataset"):
    if "time" not in obj.coords:
        return obj

    obj = obj.sortby("time")
    time_values = obj.time.values
    _, unique_idx = np.unique(time_values, return_index=True)

    n_dup = len(time_values) - len(unique_idx)
    if n_dup > 0:
        print(f"Removed {n_dup} duplicate time entries from {name}")

    return obj.isel(time=np.sort(unique_idx))


def compute_psd_from_realsht(coeffs):
    power_lm = torch.view_as_real(coeffs).pow(2).sum(dim=-1)
    psd_m0 = power_lm[..., 0]
    psd_m_pos = 2.0 * torch.sum(power_lm[..., 1:], dim=-1)
    return psd_m0 + psd_m_pos


def get_forecast_variable(ds, var_name, level_val):
    if var_name == "2m_temperature":
        da = ds[var_name]
    else:
        da = ds[var_name].sel(level=level_val)

    if var_name == "geopotential" and convert_geopotential_to_height:
        da = da / G0

    if var_name == "specific_humidity" and convert_specific_humidity_to_gkg:
        da = da * 1000.0

    return remove_duplicate_time_index(da, name=f"{var_name} forecast variable")


def compute_mean_member_psd_for_lead(fcst_all, lead_td, sht, n_times, batch_size, label):
    fcst_lead = fcst_all.sel(prediction_timedelta=lead_td)
    fcst_lead = remove_duplicate_time_index(fcst_lead, name=f"{label} lead {lead_td}")

    if "member" not in fcst_lead.dims:
        raise ValueError(
            "Forecast DataArray does not have a 'member' dimension. "
            "Cannot average spectra over ensemble members."
        )

    n_times_local = min(n_times, fcst_lead.sizes["time"])
    n_members = fcst_lead.sizes["member"]
    sum_psd = None
    count = 0

    for start_idx in range(0, n_times_local, batch_size):
        end_idx = min(start_idx + batch_size, n_times_local)
        print(f"    {label} | all {n_members} members | inits {start_idx}:{end_idx}")

        da_batch = fcst_lead.isel(time=slice(start_idx, end_idx))
        field = da_batch.transpose("time", "member", "latitude", "longitude").values
        nt, nm, nlat, nlon = field.shape
        field = field.reshape(nt * nm, nlat, nlon)

        t_field = torch.from_numpy(field).to(device=device, dtype=torch.float32)
        coeffs = sht(t_field)
        psd_batch = compute_psd_from_realsht(coeffs)
        psd_sum_batch = psd_batch.sum(dim=0)

        if sum_psd is None:
            sum_psd = torch.zeros_like(psd_sum_batch)

        sum_psd += psd_sum_batch
        count += psd_batch.shape[0]

        del field, t_field, coeffs, psd_batch, psd_sum_batch

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if count == 0:
        return None, 0

    avg_psd = (sum_psd / count).detach().cpu().numpy()
    return avg_psd, count


fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=False)

lead_td = np.timedelta64(lead_hour_to_plot, "h")
truth_lead_td = np.timedelta64(truth_lead_hour, "h")
for ax, cfg in zip(axes, plot_configs):
    var_name = cfg["var"]
    level_val = cfg["level"]
    var_label = cfg["label"]
    max_wavenumber_seen = None
    truth_plotted = False

    for case in checkpoint_zarrs:
        checkpoint_label = case["label"]
        checkpoint_color = case["color"]
        checkpoint_path = case["path"]

        print("\n" + "=" * 80)
        print(f"Processing {var_label} for {checkpoint_label}")
        print(checkpoint_path)
        print("=" * 80)

        ds_paradis = xr.open_zarr(checkpoint_path, consolidated=False)
        ds_paradis = remove_duplicate_time_index(ds_paradis, name=checkpoint_label)

        if lead_td not in ds_paradis.prediction_timedelta.values:
            print(f"Lead {lead_hour_to_plot} h not found for {checkpoint_label}. Skipping.")
            continue

        fcst_all = get_forecast_variable(ds_paradis, var_name, level_val)
        n_total_inits = fcst_all.sizes["time"]
        n_times = n_total_inits if max_inits is None else min(max_inits, n_total_inits)

        nlat = fcst_all.sizes["latitude"]
        nlon = fcst_all.sizes["longitude"]
        sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)

        if not truth_plotted:
            if truth_lead_td not in ds_paradis.prediction_timedelta.values:
                raise ValueError(
                    f"Lead {truth_lead_hour} h not found for truth/reference spectrum."
                )

            truth_psd, truth_count = compute_mean_member_psd_for_lead(
                fcst_all=fcst_all,
                lead_td=truth_lead_td,
                sht=sht,
                n_times=n_times,
                batch_size=batch_size,
                label=f"{var_label} 0 h truth/reference",
            )

            if truth_psd is None:
                raise RuntimeError("Could not compute 0 h truth/reference spectrum.")

            wavenumbers_truth = np.arange(truth_psd.shape[-1])
            max_wavenumber_seen = wavenumbers_truth[-1]

            ax.loglog(
                wavenumbers_truth[1:],
                truth_psd[1:],
                color="black",
                linestyle="--",
                lw=2.4,
                label=f"0 h truth/reference ({truth_count} samples)",
            )

            truth_plotted = True

        avg_psd, count = compute_mean_member_psd_for_lead(
            fcst_all=fcst_all,
            lead_td=lead_td,
            sht=sht,
            n_times=n_times,
            batch_size=batch_size,
            label=f"{var_label} {checkpoint_label}",
        )

        if avg_psd is None:
            print(f"No spectra computed for {checkpoint_label} at {lead_hour_to_plot} h.")
            continue

        wavenumbers = np.arange(avg_psd.shape[-1])
        max_wavenumber_seen = wavenumbers[-1]

        ax.loglog(
            wavenumbers[1:],
            avg_psd[1:],
            color=checkpoint_color,
            lw=2.5,
            label=f"{checkpoint_label} ({count} samples)",
        )

    ax.set_title(
        f"{var_label} at {lead_hour_to_plot}h",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Wave number", fontsize=18)
    ax.set_ylabel("Mean of Members Power Spectra", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    if max_wavenumber_seen is not None:
        bottom_ticks = [1, 10, 100, int(max_wavenumber_seen)]
        ax.set_xticks(bottom_ticks)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

plt.tight_layout(pad=3.0)

output_file = f"power_spectra_{lead_hour_to_plot}h_compare_noise_checkpoints.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")

print(f"Figure saved as {output_file}")