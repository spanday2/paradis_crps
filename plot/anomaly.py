import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# User settings & Constants
# ============================================================
plt.rcParams.update({'font.size': 14})
forecast_prefix="train_grf_res_3_forecast_grf_res_3"
forecast_path = f"/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/results/{forecast_prefix}.zarr"
 
truth_root = "/home/cap003/site7/datasets/era5_1deg_13level/"

G0 = 9.80665

plot_configs = [
    {"var": "geopotential",      "level": 500, "label": "Z500", "units": "m"},
    {"var": "2m_temperature",    "level": 0,   "label": "T2m",  "units": "K"},
    {"var": "specific_humidity", "level": 700, "label": "Q700", "units": "kg/kg"},
]

lead_hour = 48
lead_td = np.timedelta64(lead_hour, "h")

# Choose which initialization time to plot
time_index = 0

# Number of random members to plot
n_plot_members = 4

# ============================================================
# Helper: get truth field
# ============================================================
def get_truth_field(valid_time, var_name, level_val):
    year = valid_time.astype("datetime64[Y]").astype(int) + 1970
    year_ds = xr.open_zarr(f"{truth_root}/{year}", consolidated=False)

    if var_name == "2m_temperature":
        feature_name = "2m_temperature"
    else:
        feature_name = f"{var_name}_h{int(level_val)}"

    truth_da = year_ds["data"].sel(time=valid_time, features=feature_name)
    truth = truth_da.transpose("latitude", "longitude").values

    if var_name == "geopotential":
        truth = truth / G0

    return truth


# ============================================================
# Main plotting
# ============================================================
ds_fcst = xr.open_zarr(forecast_path)

n_members_total = ds_fcst.sizes["member"]
random_members = np.random.choice(n_members_total, n_plot_members, replace=False)

print(f"Selected random members: {random_members}")

for cfg in plot_configs:
    v_name = cfg["var"]
    v_level = cfg["level"]
    label = cfg["label"]
    units = cfg["units"]

    fcst_da = ds_fcst[v_name].sel(prediction_timedelta=lead_td)

    init_time = fcst_da["time"].isel(time=time_index).values
    valid_time = init_time + lead_td

    print(f"\nPlotting {label}")
    print(f"Init time  : {init_time}")
    print(f"Valid time : {valid_time}")
    print(f"Lead hour  : {lead_hour}")

    # -----------------------------
    # Truth
    # -----------------------------
    truth = get_truth_field(valid_time, v_name, v_level)

    # -----------------------------
    # Compute anomalies for selected members
    # -----------------------------
    anomalies = []

    for m in random_members:
        member_da = fcst_da.isel(time=time_index, member=m)

        if v_name != "2m_temperature":
            member_da = member_da.sel(level=v_level)

        member_field = member_da.transpose("latitude", "longitude").values

        if v_name == "geopotential":
            member_field = member_field / G0

        anomaly = member_field - truth
        anomalies.append(anomaly)

    anomalies = np.array(anomalies)

    # Use symmetric color limits around zero
    vmax = np.nanmax(np.abs(anomalies))
    vmin = -vmax

    levels = np.linspace(vmin, vmax, 21)

    # -----------------------------
    # Plot: 2 x 3 panels
    # -----------------------------
    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 8),
        constrained_layout=True
    )

    axes = axes.ravel()

    for i, ax in enumerate(axes):
        m = random_members[i]

        cf = ax.contourf(
            ds_fcst.longitude,
            ds_fcst.latitude,
            anomalies[i],
            levels=levels,
            extend="both",
            cmap="RdBu_r"
        )

        ax.set_title(f"Member {m} - Truth")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = fig.colorbar(
        cf,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.08
    )
    cbar.set_label(f"{label} anomaly ({units})")

    fig.suptitle(
        f"{label} anomalies: member - truth\n"
        f"Init: {init_time} | Valid: {valid_time} | Lead: {lead_hour} h",
        fontsize=16,
        fontweight="bold"
    )

    output_filename = f"{label}_anomalies_{n_plot_members}_members_lead{lead_hour}h_{forecast_prefix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_filename}")

    plt.close(fig)