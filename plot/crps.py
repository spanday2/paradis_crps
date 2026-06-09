import time
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Require properscoring for ensemble CRPS calculation
try:
    import properscoring as ps
except ImportError:
    raise ImportError(
        "The 'properscoring' package is required for this script. "
        "Please install it using: pip install properscoring"
    )


# ============================================================
# User settings
# ============================================================
plt.rcParams.update({"font.size": 14})
forecast_prefix="train_grf_res_3_forecast_grf_res_3"
forecast_path = f"/home/siw001/hall7/paradis_crps_isotropic_noise/paradis_crps/results/{forecast_prefix}.zarr"
 

plot_configs = [
    {
        "var": "geopotential",
        "level": 500,
        "label": "Z500",
        "unit": "m^2/s^2",
    },
    {
        "var": "2m_temperature",
        "level": None,
        "label": "T2m",
        "unit": "K",
    },
    {
        "var": "specific_humidity",
        "level": 700,
        "label": "Q700",
        "unit": "g/kg",
    },
]

# Index used as the verification "truth" reference. 
# The remaining members will form the forecast ensemble.
truth_member_index = 0

# Use None for all initialization times.
max_inits = None

# CHANGED: False to keep Z500 in m^2/s^2
convert_geopotential_to_height = False

# Convert specific humidity kg/kg to g/kg.
convert_specific_humidity_to_gkg = True


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
    """Sort by time and remove duplicate time coordinates."""
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


def get_forecast_variable(ds, var_name, level_val):
    """Extract, convert units, and drop duplicate times for a variable."""
    if var_name == "2m_temperature":
        da = ds[var_name]
    else:
        da = ds[var_name].sel(level=level_val)

    if var_name == "geopotential" and convert_geopotential_to_height:
        # This branch is skipped now for geopotential based on user settings
        import sys
        G0 = 9.80665
        da = da / G0

    if var_name == "specific_humidity" and convert_specific_humidity_to_gkg:
        da = da * 1000.0

    da = remove_duplicate_time_index(da, name=f"{var_name} forecast variable")
    return da


# ============================================================
# Main
# ============================================================
def main():
    script_start = time.time()

    # 1) Open dataset
    print("\nOpening forecast file...")
    print(f"  {forecast_path}")

    ds_fcst = xr.open_zarr(forecast_path, consolidated=False)
    ds_fcst = remove_duplicate_time_index(ds_fcst, name="forecast dataset")

    if "member" not in ds_fcst.dims:
        raise ValueError("Forecast dataset must have a 'member' dimension to calculate CRPS.")

    n_members = ds_fcst.sizes["member"]
    print(f"  Total ensemble members available: {n_members}")
    print(f"  Using member index {truth_member_index} as validation 'truth'.")

    # Determine subset of ensemble indices to use as the forecast distribution
    fcst_member_indices = [i for i in range(n_members) if i != truth_member_index]

    # CHANGED: Read ALL available lead times directly from the zarr file
    valid_leads = ds_fcst.prediction_timedelta.values
    valid_lead_hours = [float(lt / np.timedelta64(1, "h")) for lt in valid_leads]

    print(f"\nFound {len(valid_lead_hours)} total lead times in dataset.")
    print("Lead hours to plot:", valid_lead_hours)

    # 2) Plot configuration setup (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5), sharex=False, sharey=False)

    # 3) Process and calculate CRPS variable by variable
    for ax, cfg in zip(axes, plot_configs):
        var_start = time.time()
        var_name = cfg["var"]
        level_val = cfg["level"]
        label = cfg["label"]
        unit = cfg["unit"]

        print("\n" + "=" * 80)
        print(f"Calculating CRPS for: {label} (Units: {unit})")
        print("=" * 80)

        da_var = get_forecast_variable(ds_fcst, var_name, level_val)
        
        # Trim initialization times based on user constraints
        n_total_inits = da_var.sizes["time"]
        n_times = n_total_inits if max_inits is None else min(max_inits, n_total_inits)
        da_var = da_var.isel(time=slice(0, n_times))

        crps_vs_lead = []

        for lt in valid_leads:
            lead_hr = float(lt / np.timedelta64(1, "h"))
            print(f"  Processing lead time: {lead_hr} h...", end="\r")

            # Extract spatial/temporal block at current lead time
            da_lead = da_var.sel(prediction_timedelta=lt)
            
            # Separate the synthetic observation from the prediction ensemble
            obs = da_lead.isel(member=truth_member_index).values        # Shape: (time, lat, lon)
            ensemble = da_lead.isel(member=fcst_member_indices).values # Shape: (time, member, lat, lon)

            # Move 'member' dimension to trailing axis: (time, lat, lon, member)
            ensemble = np.moveaxis(ensemble, 1, -1)

            # Calculate CRPS point by point across all dimensions
            crps_array = ps.crps_ensemble(obs, ensemble)

            # Average across spatial dimensions (lat, lon) and initialization times
            mean_crps = np.nanmean(crps_array)
            crps_vs_lead.append(mean_crps)
        
        print(f"\n  Calculated CRPS over all lead steps successfully.")

        # Plot the CRPS curve versus Lead Time
        ax.plot(
            valid_lead_hours, 
            crps_vs_lead, 
            marker='o', 
            markersize=5, 
            lw=2.0, 
            color='tab:blue',
            label='Ensemble CRPS'
        )

        # Formatting subplots
        ax.set_title(label, fontweight="bold", pad=15)
        ax.set_xlabel("Forecast Lead Time (hours)")
        ax.set_ylabel(f"Mean CRPS ({unit})")
        ax.grid(True, which="both", alpha=0.4)
        
        var_elapsed = time.time() - var_start
        print(f"Finished {label} in {format_seconds(var_elapsed)}")

    # 4) Save Figure
    plt.tight_layout()
    output_filename = f"ensemble_crps_{forecast_prefix}_all_leads.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")

    total_elapsed = time.time() - script_start
    print("\n" + "=" * 80)
    print(f"Figure successfully saved as: {output_filename}")
    print(f"Total runtime: {format_seconds(total_elapsed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()