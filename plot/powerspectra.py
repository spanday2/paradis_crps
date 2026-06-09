# import numpy as np
# import xarray as xr
# import torch
# import torch_harmonics as th
# import matplotlib.pyplot as plt

# # ============================================================
# # User settings & Constants
# # ============================================================
# plt.rcParams.update({'font.size': 16})

# # Update these paths to your actual data locations
# forecast_zarr = "/home/shp000/site7/ensemble/paradis_crps/results/onedeg_2022_48hr.zarr"
# truth_root = "/home/cap003/site7/datasets/era5_1deg_13level/"
# G0 = 9.80665
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# plot_configs = [
#     {"var": "geopotential",      "level": 500, "label": "Z500"},
#     {"var": "2m_temperature",    "level": 0,   "label": "T2m"},
#     {"var": "specific_humidity", "level": 700, "label": "Q700"}
# ]

# lead_hour = 6
# max_inits = 10  # Set to an integer (e.g. 10) to test quickly
# lead_td = np.timedelta64(lead_hour, "h")

# # ============================================================
# # Torch-based Spectral Helper
# # ============================================================
# def compute_psd_from_realsht(coeffs: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the Power Spectral Density (PSD) from RealSHT coefficients.
#     coeffs: complex tensor of shape [..., L, M]
#     Returns: psd tensor of shape [..., L]
#     """
#     # power_lm shape: [..., L, M]
#     power_lm = torch.view_as_real(coeffs).pow(2).sum(dim=-1)
    
#     # m=0 mode
#     psd_m0 = power_lm[..., 0]
#     # m > 0 modes (doubled to account for conjugate symmetry in real signals)
#     psd_m_pos = 2.0 * torch.sum(power_lm[..., 1:], dim=-1)
    
#     return psd_m0 + psd_m_pos

# def get_truth_field(valid_time, var_name, level_val):
#     year = valid_time.astype("datetime64[Y]").astype(int) + 1970
#     year_ds = xr.open_zarr(f"{truth_root}/{year}", consolidated=False)
    
#     if var_name == "2m_temperature":
#         feature_name = "2m_temperature"
#     else:
#         feature_name = f"{var_name}_h{int(level_val)}"
        
#     truth_da = year_ds["data"].sel(time=valid_time, features=feature_name)
#     val = truth_da.transpose("latitude", "longitude").values
    
#     return val / G0 if var_name == "geopotential" else val

# # ============================================================
# # Main Processing
# # ============================================================
# ds_fcst = xr.open_zarr(forecast_zarr)
# fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# # Select two random members for individual comparison
# n_members = ds_fcst.sizes["member"]
# m1, m2 = np.random.choice(n_members, 2, replace=False)

# for ax, cfg in zip(axes, plot_configs):
#     v_name, v_level = cfg["var"], cfg["level"]
#     fcst_da = ds_fcst[v_name].sel(prediction_timedelta=lead_td)
    
#     # Initialize SHT based on grid size
#     nlat, nlon = len(ds_fcst.latitude), len(ds_fcst.longitude)
#     sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)
    
#     n_times = min(max_inits, fcst_da.sizes["time"]) if max_inits else fcst_da.sizes["time"]
    
#     sum_mean, sum_m1, sum_m2, sum_truth = 0, 0, 0, 0
#     count = 0

#     for i in range(n_times):
#         init_time = fcst_da["time"].isel(time=i).values
#         valid_time = init_time + lead_td

#         try:
#             # 1. Truth PSD
#             f_truth = get_truth_field(valid_time, v_name, v_level)
#             t_truth = torch.from_numpy(f_truth).to(device).to(torch.float32)
#             p_truth = compute_psd_from_realsht(sht(t_truth))

#             # 2. Ensemble Mean PSD
#             da_mean = fcst_da.isel(time=i).mean(dim="member")
#             if v_name != "2m_temperature": 
#                 da_mean = da_mean.sel(level=v_level)
#             f_mean = da_mean.values / G0 if v_name == "geopotential" else da_mean.values
#             t_mean = torch.from_numpy(f_mean).to(device).to(torch.float32)
#             p_mean = compute_psd_from_realsht(sht(t_mean))

#             # 3. Individual Members PSD
#             def get_member_psd(m_idx):
#                 da_m = fcst_da.isel(time=i, member=m_idx)
#                 if v_name != "2m_temperature": 
#                     da_m = da_m.sel(level=v_level)
#                 f_m = da_m.values / G0 if v_name == "geopotential" else da_m.values
#                 t_m = torch.from_numpy(f_m).to(device).to(torch.float32)
#                 return compute_psd_from_realsht(sht(t_m))

#             p_m1, p_m2 = get_member_psd(m1), get_member_psd(m2)

#             # Accumulate
#             sum_mean += p_mean
#             sum_m1 += p_m1
#             sum_m2 += p_m2
#             sum_truth += p_truth
#             count += 1
            
#         except Exception as e:
#             continue

#     if count == 0:
#         print(f"No data processed for {v_name}")
#         continue

#     # Calculate Average Absolute Power (PSD)
#     avg_truth = (sum_truth / count).cpu().numpy()
#     avg_mean  = (sum_mean / count).cpu().numpy()
#     avg_m1    = (sum_m1 / count).cpu().numpy()
#     avg_m2    = (sum_m2 / count).cpu().numpy()
    
#     degs = np.arange(avg_truth.shape[-1])

#     # Plotting: Using Log-Log for Absolute Power Spectra
#     ax.loglog(degs[1:], avg_truth[1:], label="ERA5 (Truth)", lw=2, color="black", linestyle="--")
#     ax.loglog(degs[1:], avg_mean[1:],  label="Ens. Mean",    lw=3, color="indigo")
#     ax.loglog(degs[1:], avg_m1[1:],    label=f"Mem {m1}",   lw=1.5, alpha=0.6, color="tab:blue")
#     ax.loglog(degs[1:], avg_m2[1:],    label=f"Mem {m2}",   lw=1.5, alpha=0.6, color="tab:red")

#     ax.set_title(cfg["label"], fontweight="bold")
#     ax.set_xlabel("Wavenumber (degree)")
#     if ax == axes[0]: 
#         ax.set_ylabel("Power Spectral Density")
        
#     ax.grid(True, which="both", alpha=0.3)
#     ax.legend(frameon=False, fontsize=12)

# plt.tight_layout()

# # -----------------------------
# # Save Figure
# # -----------------------------
# output_filename = "absolute_power_spectra.png"
# plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# print(f"Figure successfully saved as: {output_filename}")

# import time
# import numpy as np
# import xarray as xr
# import torch
# import torch_harmonics as th

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker


# # ============================================================
# # User settings
# # ============================================================
# plt.rcParams.update({"font.size": 14})

# forecast_path = "/home/shp000/site7/ensemble/paradis_crps/results/onedeg_2022_12hrWeight_july.zarr"

# G0 = 9.80665

# plot_configs = [
#     {
#         "var": "geopotential",
#         "level": 500,
#         "label": "Z500",
#         "unit_note": "geopotential height",
#     },
#     {
#         "var": "2m_temperature",
#         "level": None,
#         "label": "T2m",
#         "unit_note": "temperature",
#     },
#     {
#         "var": "specific_humidity",
#         "level": 700,
#         "label": "Q700",
#         "unit_note": "specific humidity",
#     },
# ]

# # Lead times to plot, in hours
# lead_hours_to_plot = [0, 24, 72, 120, 240]

# # Choose which ensemble member to plot.
# # Zero-based indexing:
# # 0 = first member
# # 1 = second member
# # ...
# ensemble_member_index = 0

# # Use None for all initialization times.
# # Use e.g. 10 for quick testing.
# max_inits = None

# # Batch size for SHT computation.
# batch_size = 8

# # Convert geopotential m^2/s^2 to geopotential height m.
# convert_geopotential_to_height = True

# # Convert specific humidity kg/kg to g/kg.
# convert_specific_humidity_to_gkg = True

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ============================================================
# # Helpers
# # ============================================================
# def format_seconds(seconds):
#     seconds = int(seconds)
#     h = seconds // 3600
#     m = (seconds % 3600) // 60
#     s = seconds % 60
#     return f"{h:02d}:{m:02d}:{s:02d}"


# def remove_duplicate_time_index(obj, name="dataset"):
#     """
#     Sort by time and remove duplicate time coordinates.

#     Keeps the first occurrence of each duplicate time.
#     Works for both xarray Dataset and DataArray.
#     """
#     if "time" not in obj.coords:
#         return obj

#     obj = obj.sortby("time")

#     time_values = obj.time.values
#     _, unique_idx = np.unique(time_values, return_index=True)

#     n_total = len(time_values)
#     n_unique = len(unique_idx)
#     n_dup = n_total - n_unique

#     if n_dup > 0:
#         print(f"Removed {n_dup} duplicate time entries from {name}")

#     return obj.isel(time=np.sort(unique_idx))


# def compute_psd_from_realsht(coeffs: torch.Tensor) -> torch.Tensor:
#     """
#     Compute power spectral density from RealSHT coefficients.

#     coeffs shape: [..., L, M]
#     returns shape: [..., L]
#     """
#     power_lm = torch.view_as_real(coeffs).pow(2).sum(dim=-1)

#     # m = 0 mode
#     psd_m0 = power_lm[..., 0]

#     # m > 0 modes doubled for real-valued fields
#     psd_m_pos = 2.0 * torch.sum(power_lm[..., 1:], dim=-1)

#     return psd_m0 + psd_m_pos


# def get_forecast_variable(ds, var_name, level_val):
#     """
#     Return forecast DataArray.

#     Expected dimensions after selection:
#         time, prediction_timedelta, member, latitude, longitude

#     Handles pressure-level and surface variables consistently.
#     """
#     if var_name == "2m_temperature":
#         da = ds[var_name]
#     else:
#         da = ds[var_name].sel(level=level_val)

#     if var_name == "geopotential" and convert_geopotential_to_height:
#         da = da / G0

#     if var_name == "specific_humidity" and convert_specific_humidity_to_gkg:
#         da = da * 1000.0

#     # Remove duplicate time index from variable data slices
#     da = remove_duplicate_time_index(da, name=f"{var_name} forecast variable")

#     return da


# def compute_average_psd_for_lead(
#     fcst_all,
#     lead_td,
#     sht,
#     n_times,
#     batch_size,
#     label,
#     ensemble_member_index,
# ):
#     """
#     For one variable and one lead time:

#     1. Select forecast at the lead time.
#     2. Select one ensemble member.
#     3. Compute PSD for each initialization time.
#     4. Average PSD over initialization times.
#     """
#     fcst_lead = fcst_all.sel(prediction_timedelta=lead_td)
#     fcst_lead = remove_duplicate_time_index(
#         fcst_lead,
#         name=f"{label} lead {lead_td}",
#     )

#     if "member" not in fcst_lead.dims:
#         raise ValueError(
#             "Forecast DataArray does not have a 'member' dimension. "
#             "Cannot select an ensemble member."
#         )

#     n_members = fcst_lead.sizes["member"]

#     if ensemble_member_index < 0 or ensemble_member_index >= n_members:
#         raise ValueError(
#             f"ensemble_member_index={ensemble_member_index} is invalid. "
#             f"Dataset has {n_members} members, valid indices are 0 to {n_members - 1}."
#         )

#     # Re-evaluate available steps locally if entries were dropped during duplicate check
#     n_times_local = min(n_times, fcst_lead.sizes["time"])

#     sum_psd = None
#     count = 0

#     n_batches = int(np.ceil(n_times_local / batch_size))

#     for batch_i, start_idx in enumerate(range(0, n_times_local, batch_size), start=1):
#         end_idx = min(start_idx + batch_size, n_times_local)
#         percent = 100.0 * batch_i / n_batches

#         print(
#             f"    {label} | member {ensemble_member_index} | "
#             f"batch {batch_i:>4}/{n_batches:<4} "
#             f"({percent:5.1f}%) | inits {start_idx}:{end_idx}"
#         )

#         da_batch = fcst_lead.isel(time=slice(start_idx, end_idx))
#         da_member = da_batch.isel(member=ensemble_member_index)
#         field = da_member.transpose("time", "latitude", "longitude").values

#         t_field = torch.from_numpy(field).to(device=device, dtype=torch.float32)
#         coeffs = sht(t_field)
#         psd_batch = compute_psd_from_realsht(coeffs)
#         psd_sum_batch = psd_batch.sum(dim=0)

#         if sum_psd is None:
#             sum_psd = torch.zeros_like(psd_sum_batch)

#         sum_psd += psd_sum_batch
#         count += psd_batch.shape[0]

#         del t_field, coeffs, psd_batch, psd_sum_batch

#         if device.type == "cuda":
#             torch.cuda.empty_cache()

#     if count == 0:
#         return None, 0

#     avg_psd = (sum_psd / count).detach().cpu().numpy()
#     return avg_psd, count


# # ============================================================
# # Main
# # ============================================================
# def main():
#     script_start = time.time()

#     print(f"Using device: {device}")
#     print(f"Using ensemble member index: {ensemble_member_index}")

#     # 1) Open forecast file
#     print("\nOpening forecast file...")
#     print(f"  {forecast_path}")

#     ds_fcst = xr.open_zarr(forecast_path, consolidated=False)
#     # Applied duplicate removal helper directly during initial setup
#     ds_fcst = remove_duplicate_time_index(ds_fcst, name="forecast dataset")

#     print("\nForecast dataset:")
#     print(f"  Number of initialization times: {ds_fcst.sizes['time']}")
#     print(f"  First initialization time: {ds_fcst.time.values[0]}")
#     print(f"  Last initialization time:  {ds_fcst.time.values[-1]}")

#     if "member" not in ds_fcst.dims:
#         raise ValueError("Forecast dataset does not have a 'member' dimension.")

#     print(f"  Number of ensemble members: {ds_fcst.sizes['member']}")

#     if ensemble_member_index < 0 or ensemble_member_index >= ds_fcst.sizes["member"]:
#         raise ValueError(
#             f"ensemble_member_index={ensemble_member_index} is invalid. "
#             f"Dataset has {ds_fcst.sizes['member']} members."
#         )

#     # 2) Grid and SHT
#     nlat = len(ds_fcst.latitude)
#     nlon = len(ds_fcst.longitude)

#     print(f"\nGrid size: nlat={nlat}, nlon={nlon}")
#     print("Initializing RealSHT...")

#     sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)

#     # 3) Determine initialization count
#     n_total_inits = ds_fcst.sizes["time"]
#     n_times = n_total_inits if max_inits is None else min(max_inits, n_total_inits)

#     print(f"\nUsing {n_times} initialization times out of {n_total_inits}")

#     # 4) Available lead times
#     available_leads = ds_fcst.prediction_timedelta.values
#     print("\nRequested lead times:")
#     print(lead_hours_to_plot)

#     available_lead_hours = [
#         float(lt / np.timedelta64(1, "h")) for lt in available_leads
#     ]
#     print("\nAvailable lead times in dataset:")
#     print(available_lead_hours)

#     # 5) Plot setup 
#     fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), sharex=True, sharey=False)

#     # 6) Loop over variables
#     for ax, cfg in zip(axes, plot_configs):
#         var_start = time.time()

#         var_name = cfg["var"]
#         level_val = cfg["level"]
#         label = cfg["label"]

#         print("\n" + "=" * 80)
#         print(f"Processing variable: {label}")
#         print("=" * 80)

#         fcst_all = get_forecast_variable(ds_fcst, var_name, level_val)
#         max_wavenumber_seen = 180 

#         for lead_i, lead_hour in enumerate(lead_hours_to_plot, start=1):
#             lead_start = time.time()
#             lead_td = np.timedelta64(lead_hour, "h")

#             if lead_td not in available_leads:
#                 print(f"\n  Lead {lead_hour} h not found. Skipping.")
#                 continue

#             print("\n" + "-" * 70)
#             print(
#                 f"  {label}: processing lead {lead_i}/{len(lead_hours_to_plot)} "
#                 f"= {lead_hour} h for member {ensemble_member_index}"
#             )
#             print("-" * 70)

#             avg_psd, count = compute_average_psd_for_lead(
#                 fcst_all=fcst_all,
#                 lead_td=lead_td,
#                 sht=sht,
#                 n_times=n_times,
#                 batch_size=batch_size,
#                 label=label,
#                 ensemble_member_index=ensemble_member_index,
#             )

#             if avg_psd is None:
#                 print(f"  No spectra computed for {label}, lead {lead_hour} h.")
#                 continue

#             wavenumbers = np.arange(avg_psd.shape[-1])
#             max_wavenumber_seen = wavenumbers[-1]

#             ax.loglog(
#                 wavenumbers[1:],
#                 avg_psd[1:],
#                 lw=1.8,
#                 label=f"{lead_hour}h",
#             )

#             lead_elapsed = time.time() - lead_start
#             print(
#                 f"  Finished {label}, lead {lead_hour} h using {count} "
#                 f"initialization times in {format_seconds(lead_elapsed)}"
#             )

#         # Style primary layout components
#         ax.set_title(label, fontweight="bold", pad=65) 
#         ax.set_xlabel("Wave number")
#         ax.set_ylabel("Power Spectral Density")
#         ax.grid(True, which="both", alpha=0.4)
#         ax.legend(frameon=False, fontsize=10)

#         # Bottom Axis: Linear layout style using plain numbers instead of power exponents
#         bottom_ticks = [1, 10, 100, int(max_wavenumber_seen)]
#         ax.set_xticks(bottom_ticks)
#         ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

#         # Top Axis: Expanded raw numbers matching vertical alignments to maximize space
#         earth_circumference_km = 2.0 * np.pi * 6371.0

#         def n_to_km(n):
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 return earth_circumference_km / n

#         def km_to_n(km):
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 return earth_circumference_km / km

#         secax = ax.secondary_xaxis('top', functions=(n_to_km, km_to_n))
#         secax.set_xlabel("Wavelength (km)", labelpad=15)
        
#         km_ticks = [40000, 10000, 5000, 2000, 1000, 500, 250]
#         secax.set_ticks(km_ticks)
#         secax.set_xticklabels([str(val) for val in km_ticks], rotation=90, verticalalignment='bottom')

#         var_elapsed = time.time() - var_start
#         print(f"\nFinished {label} in {format_seconds(var_elapsed)}")

#     # 7) Save figure
#     plt.tight_layout()

#     output_filename = f"power_spectra_member_{ensemble_member_index}_all_variables.png"
#     plt.savefig(output_filename, dpi=300, bbox_inches="tight")

#     total_elapsed = time.time() - script_start

#     print("\n" + "=" * 80)
#     print(f"Figure successfully saved as: {output_filename}")
#     print(f"Total runtime: {format_seconds(total_elapsed)}")
#     print("=" * 80)


# if __name__ == "__main__":
#     main()



### Multiple lead times for z500, t2m, q700 using all members and averaging over members and initialization times.
import time
import numpy as np
import xarray as xr
import torch
import torch_harmonics as th

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================
# User settings
# ============================================================
plt.rcParams.update({"font.size": 14})

forecast_prefix="train_grf_res_3_forecast_grf_res_3"
forecast_path = f"/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/results/{forecast_prefix}.zarr"
 

G0 = 9.80665

plot_configs = [
    {
        "var": "geopotential",
        "level": 500,
        "label": "Z500",
        "unit_note": "geopotential height",
    },
    {
        "var": "2m_temperature",
        "level": None,
        "label": "T2m",
        "unit_note": "temperature",
    },
    {
        "var": "specific_humidity",
        "level": 700,
        "label": "Q700",
        "unit_note": "specific humidity",
    },
]

# Lead times to plot, in hours
lead_hours_to_plot = [0, 24, 72, 120, 240]

# Use None for all initialization times.
# Use e.g. 10 for quick testing.
max_inits = None

# Batch size for SHT computation.
# This batches over initialization times.
# Effective SHT batch size = batch_size * number_of_members.
batch_size = 4

# Convert geopotential m^2/s^2 to geopotential height m.
convert_geopotential_to_height = True

# Convert specific humidity kg/kg to g/kg.
convert_specific_humidity_to_gkg = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Helpers
# ============================================================
def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def remove_duplicate_time_index(obj, name="dataset"):
    """
    Sort by time and remove duplicate time coordinates.

    Keeps the first occurrence of each duplicate time.
    Works for both xarray Dataset and DataArray.
    """
    if "time" not in obj.coords:
        return obj

    obj = obj.sortby("time")

    time_values = obj.time.values
    _, unique_idx = np.unique(time_values, return_index=True)

    n_total = len(time_values)
    n_unique = len(unique_idx)
    n_dup = n_total - n_unique

    if n_dup > 0:
        print(f"Removed {n_dup} duplicate time entries from {name}")

    return obj.isel(time=np.sort(unique_idx))


def compute_psd_from_realsht(coeffs: torch.Tensor) -> torch.Tensor:
    """
    Compute power spectral density from RealSHT coefficients.

    coeffs shape: [..., L, M]
    returns shape: [..., L]
    """
    power_lm = torch.view_as_real(coeffs).pow(2).sum(dim=-1)

    # m = 0 mode
    psd_m0 = power_lm[..., 0]

    # m > 0 modes doubled for real-valued fields
    psd_m_pos = 2.0 * torch.sum(power_lm[..., 1:], dim=-1)

    return psd_m0 + psd_m_pos


def get_forecast_variable(ds, var_name, level_val):
    """
    Return forecast DataArray.

    Expected dimensions after selection:
        time, prediction_timedelta, member, latitude, longitude

    Handles pressure-level and surface variables consistently.
    """
    if var_name == "2m_temperature":
        da = ds[var_name]
    else:
        da = ds[var_name].sel(level=level_val)

    if var_name == "geopotential" and convert_geopotential_to_height:
        da = da / G0

    if var_name == "specific_humidity" and convert_specific_humidity_to_gkg:
        da = da * 1000.0

    # Safety: remove duplicate time index from this DataArray too
    da = remove_duplicate_time_index(da, name=f"{var_name} forecast variable")

    return da


def compute_mean_member_psd_for_lead(
    fcst_all,
    lead_td,
    sht,
    n_times,
    batch_size,
    label,
):
    """
    For one variable and one lead time:

    1. Select forecast at the lead time.
    2. Use all ensemble members.
    3. Compute PSD separately for each member and each initialization time.
    4. Average PSD over members and initialization times.
    """
    fcst_lead = fcst_all.sel(prediction_timedelta=lead_td)
    fcst_lead = remove_duplicate_time_index(
        fcst_lead,
        name=f"{label} lead {lead_td}",
    )

    if "member" not in fcst_lead.dims:
        raise ValueError(
            "Forecast DataArray does not have a 'member' dimension. "
            "Cannot average spectra over ensemble members."
        )

    n_members = fcst_lead.sizes["member"]
    n_times_local = min(n_times, fcst_lead.sizes["time"])

    sum_psd = None
    count = 0

    n_batches = int(np.ceil(n_times_local / batch_size))

    for batch_i, start_idx in enumerate(range(0, n_times_local, batch_size), start=1):
        end_idx = min(start_idx + batch_size, n_times_local)
        percent = 100.0 * batch_i / n_batches

        print(
            f"    {label} | all {n_members} members | "
            f"batch {batch_i:>4}/{n_batches:<4} "
            f"({percent:5.1f}%) | inits {start_idx}:{end_idx}"
        )

        da_batch = fcst_lead.isel(time=slice(start_idx, end_idx))
        field = da_batch.transpose(
            "time", "member", "latitude", "longitude"
        ).values

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


# ============================================================
# Main
# ============================================================
def main():
    script_start = time.time()

    print(f"Using device: {device}")

    # 1) Open forecast file
    print("\nOpening forecast file...")
    print(f"  {forecast_path}")

    ds_fcst = xr.open_zarr(forecast_path, consolidated=False)
    ds_fcst = remove_duplicate_time_index(ds_fcst, name="forecast dataset")

    print("\nForecast dataset after duplicate removal:")
    print(f"  Number of initialization times: {ds_fcst.sizes['time']}")
    print(f"  First initialization time: {ds_fcst.time.values[0]}")
    print(f"  Last initialization time:  {ds_fcst.time.values[-1]}")

    if "member" not in ds_fcst.dims:
        raise ValueError("Forecast dataset does not have a 'member' dimension.")

    print(f"  Number of ensemble members: {ds_fcst.sizes['member']}")

    # 2) Grid and SHT
    nlat = len(ds_fcst.latitude)
    nlon = len(ds_fcst.longitude)

    print(f"\nGrid size: nlat={nlat}, nlon={nlon}")
    print("Initializing RealSHT...")

    sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)

    # 3) Determine initialization count
    n_total_inits = ds_fcst.sizes["time"]
    n_times = n_total_inits if max_inits is None else min(max_inits, n_total_inits)

    print(f"\nUsing {n_times} initialization times out of {n_total_inits}")
    print(f"Using all {ds_fcst.sizes['member']} ensemble members")

    # 4) Available lead times
    available_leads = ds_fcst.prediction_timedelta.values
    print("\nRequested lead times:")
    print(lead_hours_to_plot)

    available_lead_hours = [
        float(lt / np.timedelta64(1, "h")) for lt in available_leads
    ]
    print("\nAvailable lead times in dataset:")
    print(available_lead_hours)

    # 5) Plot setup
    # Increased vertical size slightly to accommodate the vertical top labels cleanly
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), sharex=True, sharey=False)

    # 6) Loop over variables
    for ax, cfg in zip(axes, plot_configs):
        var_start = time.time()

        var_name = cfg["var"]
        level_val = cfg["level"]
        label = cfg["label"]

        print("\n" + "=" * 80)
        print(f"Processing variable: {label}")
        print("=" * 80)

        fcst_all = get_forecast_variable(ds_fcst, var_name, level_val)
        max_wavenumber_seen = 180 

        for lead_i, lead_hour in enumerate(lead_hours_to_plot, start=1):
            lead_start = time.time()
            lead_td = np.timedelta64(lead_hour, "h")

            if_lead_not_found = lead_td not in available_leads
            if if_lead_not_found:
                print(f"\n  Lead {lead_hour} h not found. Skipping.")
                continue

            print("\n" + "-" * 70)
            print(
                f"  {label}: processing lead {lead_i}/{len(lead_hours_to_plot)} "
                f"= {lead_hour} h using all members"
            )
            print("-" * 70)

            avg_psd, count = compute_mean_member_psd_for_lead(
                fcst_all=fcst_all,
                lead_td=lead_td,
                sht=sht,
                n_times=n_times,
                batch_size=batch_size,
                label=label,
            )

            if avg_psd is None:
                print(f"  No spectra computed for {label}, lead {lead_hour} h.")
                continue

            wavenumbers = np.arange(avg_psd.shape[-1])
            max_wavenumber_seen = wavenumbers[-1]

            ax.loglog(
                wavenumbers[1:],
                avg_psd[1:],
                lw=1.8,
                label=f"{lead_hour}h",
            )

            lead_elapsed = time.time() - lead_start
            print(
                f"  Finished {label}, lead {lead_hour} h using {count} "
                f"time-member samples in {format_seconds(lead_elapsed)}"
            )

        # Style primary axes
        # Padded title up to 65 to make ample room for the expanded vertical top numbers
        ax.set_title(label, fontweight="bold", pad=65) 
        ax.set_xlabel("Wave number")
        ax.set_ylabel("Mean of Members Power Spectra")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(frameon=False, fontsize=10)

        # ------------------------------------------------------------
        # Bottom Axis: Scalar numbers & explicit ending tick 
        # ------------------------------------------------------------
        bottom_ticks = [1, 10, 100, int(max_wavenumber_seen)]
        ax.set_xticks(bottom_ticks)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        # ------------------------------------------------------------
        # Top Axis: Expanded full numeric values & Vertical rotation
        # ------------------------------------------------------------
        earth_circumference_km = 2.0 * np.pi * 6371.0

        def n_to_km(n):
            with np.errstate(divide='ignore', invalid='ignore'):
                return earth_circumference_km / n

        def km_to_n(km):
            with np.errstate(divide='ignore', invalid='ignore'):
                return earth_circumference_km / km

        secax = ax.secondary_xaxis('top', functions=(n_to_km, km_to_n))
        secax.set_xlabel("Wavelength (km)", labelpad=15)
        
        km_ticks = [40000, 10000, 5000, 2000, 1000, 500, 250]
        secax.set_ticks(km_ticks)
        
        # Placed explicit raw numbers and added rotation=90 to clear overlaps
        secax.set_xticklabels([str(val) for val in km_ticks], rotation=90, verticalalignment='bottom')

        var_elapsed = time.time() - var_start
        print(f"\nFinished {label} in {format_seconds(var_elapsed)}")

    # 7) Save figure
    plt.tight_layout()

    output_filename = f"power_spectra_mean_over_member_spectra_all_variables_{forecast_prefix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")

    total_elapsed = time.time() - script_start

    print("\n" + "=" * 80)
    print(f"Figure successfully saved as: {output_filename}")
    print(f"Total runtime: {format_seconds(total_elapsed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# import time
# import numpy as np
# import xarray as xr
# import torch
# import torch_harmonics as th

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker


# # ============================================================
# # User settings
# # ============================================================
# plt.rcParams.update({"font.size": 14})

# # ------------------------------------------------------------
# # Three forecast Zarr files from three different checkpoints
# # Replace these paths with your actual files
# # ------------------------------------------------------------
# forecast_cases = [
#     {
#         "label": "100K",
#         "path": "/home/shp000/site7/ensemble/paradis_crps/results/100K_forecast/onedeg_100K_2022_july.zarr",
#     },
#     {
#         "label": "200K",
#         "path": "/home/shp000/site7/ensemble/paradis_crps/results/200K_forecast/onedeg_200K_2022_july.zarr",
#     },
#     {
#         "label": "300K",
#         "path": "/home/shp000/site7/ensemble/paradis_crps/results/onedeg_300K_2022_july.zarr",
#     },
# ]

# # Use 0 h reference/truth from this file.
# # Usually choose one of the above results.
# truth_reference_path = forecast_cases[0]["path"]

# # Z500 only
# var_name = "geopotential"
# level_val = 500
# var_label = "Z500"

# G0 = 9.80665
# convert_geopotential_to_height = True

# # Columns: lead times to compare
# lead_hours_to_plot = [6, 120, 240]

# # Reference/truth line lead time
# truth_lead_hour = 0

# # Use None for all initialization times.
# # Use e.g. 10 for quick testing.
# max_inits = None

# # Batch size for SHT computation.
# # Effective SHT batch size = batch_size * number_of_members
# batch_size = 4

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ============================================================
# # Helpers
# # ============================================================
# def format_seconds(seconds):
#     seconds = int(seconds)
#     h = seconds // 3600
#     m = (seconds % 3600) // 60
#     s = seconds % 60
#     return f"{h:02d}:{m:02d}:{s:02d}"


# def remove_duplicate_time_index(obj, name="dataset"):
#     """
#     Sort by time and remove duplicate time coordinates.

#     Keeps the first occurrence of each duplicate time.
#     Works for both xarray Dataset and DataArray.
#     """
#     if "time" not in obj.coords:
#         return obj

#     obj = obj.sortby("time")

#     time_values = obj.time.values
#     _, unique_idx = np.unique(time_values, return_index=True)

#     n_total = len(time_values)
#     n_unique = len(unique_idx)
#     n_dup = n_total - n_unique

#     if n_dup > 0:
#         print(f"Removed {n_dup} duplicate time entries from {name}")

#     return obj.isel(time=np.sort(unique_idx))


# def compute_psd_from_realsht(coeffs: torch.Tensor) -> torch.Tensor:
#     """
#     Compute power spectral density from RealSHT coefficients.

#     coeffs shape: [..., L, M]
#     returns shape: [..., L]
#     """
#     power_lm = torch.view_as_real(coeffs).pow(2).sum(dim=-1)

#     # m = 0 mode
#     psd_m0 = power_lm[..., 0]

#     # m > 0 modes doubled for real-valued fields
#     psd_m_pos = 2.0 * torch.sum(power_lm[..., 1:], dim=-1)

#     return psd_m0 + psd_m_pos


# def get_z500_da(ds):
#     """
#     Return Z500 forecast/reference DataArray.

#     Expected dimensions after selection:
#         time, prediction_timedelta, member, latitude, longitude

#     If convert_geopotential_to_height=True:
#         converts geopotential [m^2/s^2] to geopotential height [m].
#     """
#     da = ds[var_name].sel(level=level_val)

#     if convert_geopotential_to_height:
#         da = da / G0

#     da = remove_duplicate_time_index(da, name="Z500 DataArray")
#     return da


# def compute_mean_psd_for_lead(
#     da_all,
#     lead_td,
#     sht,
#     n_times,
#     batch_size,
#     label,
# ):
#     """
#     Compute mean power spectrum for one DataArray and one lead time.

#     Handles both:
#       1. ensemble forecast with member dimension:
#             time, prediction_timedelta, member, latitude, longitude

#       2. single reference field without member dimension:
#             time, prediction_timedelta, latitude, longitude

#     It averages PSD over all selected initialization times and members.
#     """
#     da_lead = da_all.sel(prediction_timedelta=lead_td)
#     da_lead = remove_duplicate_time_index(
#         da_lead,
#         name=f"{label} lead {lead_td}",
#     )

#     n_times_local = min(n_times, da_lead.sizes["time"])

#     has_member = "member" in da_lead.dims
#     n_members = da_lead.sizes["member"] if has_member else 1

#     sum_psd = None
#     count = 0

#     n_batches = int(np.ceil(n_times_local / batch_size))

#     for batch_i, start_idx in enumerate(range(0, n_times_local, batch_size), start=1):
#         end_idx = min(start_idx + batch_size, n_times_local)
#         percent = 100.0 * batch_i / n_batches

#         print(
#             f"    {label} | lead={lead_td} | "
#             f"batch {batch_i:>4}/{n_batches:<4} "
#             f"({percent:5.1f}%) | inits {start_idx}:{end_idx}"
#         )

#         da_batch = da_lead.isel(time=slice(start_idx, end_idx))

#         if has_member:
#             field = da_batch.transpose(
#                 "time", "member", "latitude", "longitude"
#             ).values

#             nt, nm, nlat, nlon = field.shape
#             field = field.reshape(nt * nm, nlat, nlon)

#         else:
#             field = da_batch.transpose(
#                 "time", "latitude", "longitude"
#             ).values

#             nt, nlat, nlon = field.shape
#             field = field.reshape(nt, nlat, nlon)

#         t_field = torch.from_numpy(field).to(device=device, dtype=torch.float32)

#         coeffs = sht(t_field)
#         psd_batch = compute_psd_from_realsht(coeffs)
#         psd_sum_batch = psd_batch.sum(dim=0)

#         if sum_psd is None:
#             sum_psd = torch.zeros_like(psd_sum_batch)

#         sum_psd += psd_sum_batch
#         count += psd_batch.shape[0]

#         del field, t_field, coeffs, psd_batch, psd_sum_batch

#         if device.type == "cuda":
#             torch.cuda.empty_cache()

#     if count == 0:
#         return None, 0

#     avg_psd = (sum_psd / count).detach().cpu().numpy()
#     return avg_psd, count


# def get_available_lead_hours(ds):
#     return [
#         float(lt / np.timedelta64(1, "h"))
#         for lt in ds.prediction_timedelta.values
#     ]


# # ============================================================
# # Main
# # ============================================================
# def main():
#     script_start = time.time()

#     print(f"Using device: {device}")

#     # ------------------------------------------------------------
#     # Open all forecast datasets
#     # ------------------------------------------------------------
#     datasets = []

#     for case in forecast_cases:
#         print("\nOpening forecast file:")
#         print(f"  {case['label']}: {case['path']}")

#         ds = xr.open_zarr(case["path"], consolidated=False)
#         ds = remove_duplicate_time_index(ds, name=case["label"])

#         datasets.append(
#             {
#                 "label": case["label"],
#                 "path": case["path"],
#                 "ds": ds,
#                 "z500": get_z500_da(ds),
#             }
#         )

#         print(f"  Number of initialization times: {ds.sizes['time']}")
#         print(f"  First initialization time: {ds.time.values[0]}")
#         print(f"  Last initialization time:  {ds.time.values[-1]}")

#         if "member" in ds.dims:
#             print(f"  Number of ensemble members: {ds.sizes['member']}")

#         print(f"  Available leads: {get_available_lead_hours(ds)}")

#     # ------------------------------------------------------------
#     # Open reference/truth dataset
#     # ------------------------------------------------------------
#     print("\nOpening 0 h reference/truth file:")
#     print(f"  {truth_reference_path}")

#     ds_truth_ref = xr.open_zarr(truth_reference_path, consolidated=False)
#     ds_truth_ref = remove_duplicate_time_index(ds_truth_ref, name="truth/reference dataset")
#     z500_truth_ref = get_z500_da(ds_truth_ref)

#     # ------------------------------------------------------------
#     # Grid and SHT
#     # ------------------------------------------------------------
#     nlat = len(datasets[0]["ds"].latitude)
#     nlon = len(datasets[0]["ds"].longitude)

#     print(f"\nGrid size: nlat={nlat}, nlon={nlon}")
#     print("Initializing RealSHT...")

#     sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)

#     # ------------------------------------------------------------
#     # Determine initialization count
#     # ------------------------------------------------------------
#     n_total_inits = datasets[0]["ds"].sizes["time"]
#     n_times = n_total_inits if max_inits is None else min(max_inits, n_total_inits)

#     print(f"\nUsing {n_times} initialization times out of {n_total_inits}")

#     # ------------------------------------------------------------
#     # Compute 0 h reference/truth spectrum once
#     # ------------------------------------------------------------
#     truth_lead_td = np.timedelta64(truth_lead_hour, "h")

#     if truth_lead_td not in ds_truth_ref.prediction_timedelta.values:
#         raise ValueError(
#             f"Requested truth/reference lead {truth_lead_hour} h not found. "
#             f"Available leads are: {get_available_lead_hours(ds_truth_ref)}"
#         )

#     print("\nComputing 0 h reference/truth spectrum...")

#     truth_psd, truth_count = compute_mean_psd_for_lead(
#         da_all=z500_truth_ref,
#         lead_td=truth_lead_td,
#         sht=sht,
#         n_times=n_times,
#         batch_size=batch_size,
#         label=f"0 h reference/truth",
#     )

#     if truth_psd is None:
#         raise RuntimeError("Could not compute 0 h reference/truth spectrum.")

#     print(
#         f"Finished 0 h reference/truth using {truth_count} samples."
#     )

#     # ------------------------------------------------------------
#     # Plot: 1 row x 3 columns
#     # ------------------------------------------------------------
#     fig, axes = plt.subplots(
#         1,
#         3,
#         figsize=(24, 7.5),
#         sharex=True,
#         sharey=True,
#     )

#     max_wavenumber_seen = truth_psd.shape[-1] - 1
#     wavenumbers_truth = np.arange(truth_psd.shape[-1])

#     # ------------------------------------------------------------
#     # Loop over lead times / columns
#     # ------------------------------------------------------------
#     for ax, lead_hour in zip(axes, lead_hours_to_plot):
#         lead_start = time.time()
#         lead_td = np.timedelta64(lead_hour, "h")

#         print("\n" + "=" * 80)
#         print(f"Processing lead {lead_hour} h")
#         print("=" * 80)

#         # Plot 0 h reference/truth line
#         ax.loglog(
#             wavenumbers_truth[1:],
#             truth_psd[1:],
#             lw=2.4,
#             color="black",
#             linestyle="--",
#             label="0 h truth/reference",
#         )

#         # Plot each checkpoint result
#         for case in datasets:
#             label = case["label"]
#             ds = case["ds"]
#             z500 = case["z500"]

#             available_leads = ds.prediction_timedelta.values

#             if lead_td not in available_leads:
#                 print(f"  {label}: lead {lead_hour} h not found. Skipping.")
#                 continue

#             print(f"\n  Computing {label}, lead {lead_hour} h")

#             avg_psd, count = compute_mean_psd_for_lead(
#                 da_all=z500,
#                 lead_td=lead_td,
#                 sht=sht,
#                 n_times=n_times,
#                 batch_size=batch_size,
#                 label=f"{label}",
#             )

#             if avg_psd is None:
#                 print(f"  No spectra computed for {label}, lead {lead_hour} h.")
#                 continue

#             wavenumbers = np.arange(avg_psd.shape[-1])
#             max_wavenumber_seen = max(max_wavenumber_seen, wavenumbers[-1])

#             ax.loglog(
#                 wavenumbers[1:],
#                 avg_psd[1:],
#                 lw=2.0,
#                 label=label,
#             )

#             print(
#                 f"  Finished {label}, lead {lead_hour} h using {count} "
#                 f"time-member samples."
#             )

#         # ------------------------------------------------------------
#         # Axes style
#         # ------------------------------------------------------------
#         ax.set_title(f"{var_label}, lead {lead_hour} h", fontweight="bold", pad=65)
#         ax.set_xlabel("Wave number")
#         ax.grid(True, which="both", alpha=0.4)
#         ax.legend(frameon=False, fontsize=10)

#         lead_elapsed = time.time() - lead_start
#         print(f"Finished lead {lead_hour} h in {format_seconds(lead_elapsed)}")

#     axes[0].set_ylabel("Mean Power Spectrum")

#     # ------------------------------------------------------------
#     # Axis ticks and top wavelength axis
#     # ------------------------------------------------------------
#     earth_circumference_km = 2.0 * np.pi * 6371.0

#     def n_to_km(n):
#         with np.errstate(divide="ignore", invalid="ignore"):
#             return earth_circumference_km / n

#     def km_to_n(km):
#         with np.errstate(divide="ignore", invalid="ignore"):
#             return earth_circumference_km / km

#     for ax in axes:
#         bottom_ticks = [1, 10, 100, int(max_wavenumber_seen)]
#         ax.set_xticks(bottom_ticks)
#         ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

#         secax = ax.secondary_xaxis("top", functions=(n_to_km, km_to_n))
#         secax.set_xlabel("Wavelength (km)", labelpad=15)

#         km_ticks = [40000, 10000, 5000, 2000, 1000, 500, 250]
#         secax.set_ticks(km_ticks)
#         secax.set_xticklabels(
#             [str(val) for val in km_ticks],
#             rotation=90,
#             verticalalignment="bottom",
#         )

#     plt.tight_layout()

#     output_filename = "z500_power_spectra_three_checkpoints_6h_120h_240h.png"
#     plt.savefig(output_filename, dpi=300, bbox_inches="tight")

#     total_elapsed = time.time() - script_start

#     print("\n" + "=" * 80)
#     print(f"Figure successfully saved as: {output_filename}")
#     print(f"Total runtime: {format_seconds(total_elapsed)}")
#     print("=" * 80)


# if __name__ == "__main__":
#     main()