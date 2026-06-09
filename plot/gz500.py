# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# import dask

# # -----------------------------
# # 1) Global Plot & Config
# # -----------------------------
# plt.rcParams.update({'font.size': 16}) 

# # chunks={} ensures we use Dask to avoid Memory Exhaustion (OOM)
# forecast_prefix="train_grf_noise_forecast_raw"
# ds_paradis = xr.open_zarr("/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/results/"+forecast_prefix+"_10k_ckpt.zarr", consolidated=False, chunks={})
# base_root = "/home/cap003/site7/datasets/era5_1deg_13level/"

# plot_configs = [
#     {"var": "geopotential",      "level": 500, "label": "Z500", "units": r"m$^2$ s$^{-2}$"},
#     {"var": "specific_humidity", "level": 700, "label": "Q700", "units": "g/kg"},
#     {"var": "2m_temperature",    "level": 0,   "label": "T2m",  "units": "K"}
# ]

# # -----------------------------
# # 2) Helpers
# # -----------------------------
# year_cache = {}

# def open_year_ds(year):
#     if year not in year_cache:
#         year_path = f"{base_root}/{year}"
#         year_cache[year] = xr.open_zarr(year_path, consolidated=False, chunks={})
#     return year_cache[year]

# def get_truth_slice(year_ds, var_name, level_val, times):
#     feature_name = "2m_temperature" if var_name == "2m_temperature" else f"{var_name}_h{int(level_val)}"
#     return year_ds["data"].sel(time=times, features=feature_name)

# # -----------------------------
# # 3) Setup Figure
# # -----------------------------
# fig, axes = plt.subplots(1, 3, figsize=(24, 8))
# lat = ds_paradis.latitude
# weights_da = xr.DataArray(np.cos(np.deg2rad(lat.values)), coords={"latitude": lat}, dims=("latitude",))
# lead_times = ds_paradis.prediction_timedelta.values[1:]

# # -----------------------------
# # 4) Main Processing Loop
# # -----------------------------
# for ax, cfg in zip(axes, plot_configs):
#     v_name, v_level = cfg["var"], cfg["level"]
#     rmse_list, spread_list, lead_hours_used = [], [], []

#     print(f"Processing {cfg['label']}...")

#     for lt in lead_times:
#         # Lazy selection
#         if v_name == "2m_temperature":
#             fct = ds_paradis[v_name].sel(prediction_timedelta=lt)
#         else:
#             fct = ds_paradis[v_name].sel(level=v_level, prediction_timedelta=lt)

#         # Division by G0 removed here to keep Geopotential

#         target_times = fct.time + lt
#         years = np.array([t.astype("datetime64[Y]").astype(int) + 1970 for t in target_times.values])
        
#         obs_parts, fct_parts = [], []

#         for year in np.unique(years):
#             mask = years == year
#             year_ds = open_year_ds(int(year))
#             common_targets = np.intersect1d(target_times.values[mask], year_ds.time.values)
            
#             if common_targets.size == 0: continue

#             obs_year = get_truth_slice(year_ds, v_name, v_level, common_targets)
            
#             # Division by G0 removed here as well

#             fct_year = fct.sel(time=common_targets - lt)
#             fct_year = fct_year.assign_coords(time=(fct_year.time + lt))

#             obs_parts.append(obs_year)
#             fct_parts.append(fct_year)

#         if not obs_parts: continue

#         obs = xr.concat(obs_parts, dim="time")
#         fct_ens = xr.concat(fct_parts, dim="time")

#         # Define Lazy Metrics
#         ens_mean = fct_ens.mean(dim="member")
#         mse_lat = ((ens_mean - obs) ** 2).mean(dim=["time", "longitude"])
#         global_rmse_lazy = np.sqrt((mse_lat * weights_da).sum() / weights_da.sum())

#         R = fct_ens.sizes["member"]
#         var_lat = fct_ens.var(dim="member", ddof=1).mean(dim=["time", "longitude"])
#         global_spread_lazy = np.sqrt(((R + 1) / R) * (var_lat * weights_da).sum() / weights_da.sum())

#         # Trigger computation for final scalars only
#         rmse_val, spread_val = dask.compute(global_rmse_lazy, global_spread_lazy)

#         rmse_list.append(float(rmse_val))
#         spread_list.append(float(spread_val))
#         lead_hours_used.append(float(lt / np.timedelta64(1, "h")))

#     # -----------------------------
#     # 5) Plotting & Formatting
#     # -----------------------------
#     ax.plot(lead_hours_used, rmse_list, color="indigo", marker="o", lw=3, markersize=8, label="Mean RMSE")
#     ax.plot(lead_hours_used, spread_list, color="blue", marker="o", lw=3, markersize=8, linestyle=":", label="Spread")
    
#     ax.set_title(cfg["label"], fontsize=24, fontweight='bold', pad=20)
#     ax.set_xlabel("Lead time (hours)", fontsize=20)
#     ax.set_ylabel(f" {cfg['units']}", fontsize=20)
#     ax.tick_params(labelsize=18)
#     ax.legend(fontsize=18, frameon=False)
#     ax.grid(True, alpha=0.3)
    
#     if rmse_list:
#         ax.set_ylim(0, max(rmse_list) * 1.2)

# plt.tight_layout(pad=3.0)

# # -----------------------------
# # 6) Save Figure
# # -----------------------------
# output_file = f"forecast_metrics_comparison_geopotential_2_{forecast_prefix}.png"
# plt.savefig(output_file, dpi=300, bbox_inches='tight')
# print(f"Figure saved as {output_file}")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask

# -----------------------------
# 1) Global Plot & Config
# -----------------------------
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

base_root = "/home/cap003/site7/datasets/era5_1deg_13level/"

# Z500 only
v_name = "geopotential"
v_level = 500
var_label = "Z500"
units = r"m$^2$ s$^{-2}$"

# If you want geopotential height in m instead of geopotential:
convert_geopotential_to_height = False
G0 = 9.80665

# -----------------------------
# 2) Helpers
# -----------------------------
year_cache = {}

def open_year_ds(year):
    if year not in year_cache:
        year_path = f"{base_root}/{year}"
        year_cache[year] = xr.open_zarr(
            year_path,
            consolidated=False,
            chunks={},
        )
    return year_cache[year]


def get_truth_slice(year_ds, var_name, level_val, times):
    feature_name = f"{var_name}_h{int(level_val)}"
    obs = year_ds["data"].sel(time=times, features=feature_name)

    if var_name == "geopotential" and convert_geopotential_to_height:
        obs = obs / G0

    return obs


def get_forecast_z500(ds, lead_time):
    fct = ds[v_name].sel(level=v_level, prediction_timedelta=lead_time)

    if convert_geopotential_to_height:
        fct = fct / G0

    return fct


def compute_rmse_spread_for_checkpoint(ds_paradis, label):
    lat = ds_paradis.latitude

    weights_da = xr.DataArray(
        np.cos(np.deg2rad(lat.values)),
        coords={"latitude": lat},
        dims=("latitude",),
    )

    max_lead_time = np.timedelta64(72, "h")
    lead_times = ds_paradis.prediction_timedelta.values[1:]
    lead_times = lead_times[lead_times <= max_lead_time]

    rmse_lazy_list = []
    spread_lazy_list = []
    lead_hours_used = []

    print(f"Processing {label}...")

    for lt in lead_times:
        fct = get_forecast_z500(ds_paradis, lt)

        target_times = fct.time + lt

        years = np.array([
            t.astype("datetime64[Y]").astype(int) + 1970
            for t in target_times.values
        ])

        obs_parts = []
        fct_parts = []

        for year in np.unique(years):
            mask = years == year
            year_ds = open_year_ds(int(year))

            common_targets = np.intersect1d(
                target_times.values[mask],
                year_ds.time.values,
            )

            if common_targets.size == 0:
                continue

            obs_year = get_truth_slice(
                year_ds,
                v_name,
                v_level,
                common_targets,
            )

            fct_year = fct.sel(time=common_targets - lt)
            fct_year = fct_year.assign_coords(time=(fct_year.time + lt))

            obs_parts.append(obs_year)
            fct_parts.append(fct_year)

        if not obs_parts:
            continue

        obs = xr.concat(obs_parts, dim="time")
        fct_ens = xr.concat(fct_parts, dim="time")

        # Ensemble mean RMSE
        ens_mean = fct_ens.mean(dim="member")

        mse_lat = ((ens_mean - obs) ** 2).mean(dim=["time", "longitude"])

        global_rmse_lazy = np.sqrt(
            (mse_lat * weights_da).sum(dim="latitude")
            / weights_da.sum(dim="latitude")
        )

        # Ensemble spread with Fortin correction
        R = fct_ens.sizes["member"]

        var_lat = fct_ens.var(dim="member", ddof=1).mean(
            dim=["time", "longitude"]
        )

        global_spread_lazy = np.sqrt(
            ((R + 1) / R)
            * (var_lat * weights_da).sum(dim="latitude")
            / weights_da.sum(dim="latitude")
        )

        rmse_lazy_list.append(global_rmse_lazy)
        spread_lazy_list.append(global_spread_lazy)
        lead_hours_used.append(float(lt / np.timedelta64(1, "h")))

    if len(rmse_lazy_list) == 0:
        return [], [], []

    print(f"Computing metrics for {label} in one Dask call...")

    computed = dask.compute(*(rmse_lazy_list + spread_lazy_list))

    n = len(rmse_lazy_list)

    rmse_list = [float(x) for x in computed[:n]]
    spread_list = [float(x) for x in computed[n:]]

    return lead_hours_used, rmse_list, spread_list


# -----------------------------
# 3) Setup Figure
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(24, 8), sharey=True)

# -----------------------------
# 4) Main Processing Loop
# -----------------------------
all_rmse_values = []
all_spread_values = []

for case in checkpoint_zarrs:
    checkpoint_label = case["label"]
    checkpoint_color = case["color"]
    checkpoint_path = case["path"]

    print("\n" + "=" * 80)
    print(f"Opening {checkpoint_label}")
    print(checkpoint_path)
    print("=" * 80)

    ds_paradis = xr.open_zarr(
        checkpoint_path,
        consolidated=False,
        chunks={},
    )

    lead_hours_used, rmse_list, spread_list = compute_rmse_spread_for_checkpoint(
        ds_paradis,
        checkpoint_label,
    )

    ax.plot(
        lead_hours_used,
        rmse_list,
        color=checkpoint_color,
        marker="o",
        lw=3,
        markersize=8,
        label=f"{checkpoint_label} RMSE",
    )

    all_rmse_values.extend(rmse_list)
    all_spread_values.extend(spread_list)

    # ax.plot(
    #     lead_hours_used,
    #     spread_list,
    #     color="blue",
    #     marker="o",
    #     lw=3,
    #     markersize=8,
    #     linestyle=":",
    #     label="Spread",
    # )

ax.set_title(
    var_label,
    fontsize=24,
    fontweight="bold",
    pad=20,
)

ax.set_xlabel("Lead time (hours)", fontsize=20)
ax.set_ylabel(units, fontsize=20)
ax.tick_params(labelsize=18)
ax.legend(fontsize=18, frameon=False)
ax.grid(True, alpha=0.3)

if all_rmse_values or all_spread_values:
    ymax = max(all_rmse_values + all_spread_values)
    ax.set_ylim(0, ymax * 1.2)

plt.tight_layout(pad=3.0)

# -----------------------------
# 5) Save Figure
# -----------------------------
output_file = "z500_rmse_compare_noise_checkpoints.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")

print(f"Figure saved as {output_file}")
