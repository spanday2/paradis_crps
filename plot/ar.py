import numpy as np
import xarray as xr
import torch
import torch_harmonics as th
import matplotlib.pyplot as plt

# ============================================================
# User settings & Constants
# ============================================================
plt.rcParams.update({'font.size': 16})
forecast_zarr = "/home/shp000/site7/ensemble/paradis_crps/results/onedeg_2022_48hr.zarr"
truth_root = "/home/cap003/site7/datasets/era5_1deg_13level/"
G0 = 9.80665
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plot_configs = [
    {"var": "geopotential",      "level": 500, "label": "Z500"},
    {"var": "2m_temperature",    "level": 0,   "label": "T2m"},
    {"var": "specific_humidity", "level": 700, "label": "Q700"}
]

lead_hour = 48
max_inits = None 
lead_td = np.timedelta64(lead_hour, "h")

# ============================================================
# Torch-based Spectral Helper
# ============================================================
def compute_psd_from_realsht(coeffs: torch.Tensor) -> torch.Tensor:
    """PSD from RealSHT coeffs. coeffs: complex [...,L,M] -> psd [...,L]."""
    # power_lm shape: [..., L, M, 2] -> sum over real/imag parts
    power_lm = torch.view_as_real(coeffs).pow(2).sum(dim=-1)
    psd_m0 = power_lm[..., 0]
    psd_m_pos = 2.0 * torch.sum(power_lm[..., 1:], dim=-1)
    return psd_m0 + psd_m_pos

def get_truth_field(valid_time, var_name, level_val):
    year = valid_time.astype("datetime64[Y]").astype(int) + 1970
    year_ds = xr.open_zarr(f"{truth_root}/{year}", consolidated=False)
    feature_name = "2m_temperature" if var_name == "2m_temperature" else f"{var_name}_h{int(level_val)}"
    truth_da = year_ds["data"].sel(time=valid_time, features=feature_name)
    val = truth_da.transpose("latitude", "longitude").values
    return val / G0 if var_name == "geopotential" else val

# ============================================================
# Main Processing
# ============================================================
ds_fcst = xr.open_zarr(forecast_zarr)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Select members
n_members = ds_fcst.sizes["member"]
m1, m2 = np.random.choice(n_members, 2, replace=False)

for ax, cfg in zip(axes, plot_configs):
    v_name, v_level = cfg["var"], cfg["level"]
    fcst_da = ds_fcst[v_name].sel(prediction_timedelta=lead_td)
    
    # Initialize SHT based on grid size
    nlat, nlon = len(ds_fcst.latitude), len(ds_fcst.longitude)
    sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)
    
    n_times = min(max_inits, fcst_da.sizes["time"]) if max_inits else fcst_da.sizes["time"]
    
    sum_mean, sum_m1, sum_m2, sum_truth = 0, 0, 0, 0
    count = 0

    for i in range(n_times):
        init_time = fcst_da["time"].isel(time=i).values
        valid_time = init_time + lead_td

        try:
            # 1. Truth
            f_truth = get_truth_field(valid_time, v_name, v_level)
            t_truth = torch.from_numpy(f_truth).to(device).to(torch.float32)
            p_truth = compute_psd_from_realsht(sht(t_truth))

            # 2. Ensemble Mean
            da_mean = fcst_da.isel(time=i).mean(dim="member")
            if v_name != "2m_temperature": da_mean = da_mean.sel(level=v_level)
            f_mean = da_mean.values / G0 if v_name == "geopotential" else da_mean.values
            t_mean = torch.from_numpy(f_mean).to(device).to(torch.float32)
            p_mean = compute_psd_from_realsht(sht(t_mean))

            # 3. Members
            def get_member_psd(m_idx):
                da_m = fcst_da.isel(time=i, member=m_idx)
                if v_name != "2m_temperature": da_m = da_m.sel(level=v_level)
                f_m = da_m.values / G0 if v_name == "geopotential" else da_m.values
                t_m = torch.from_numpy(f_m).to(device).to(torch.float32)
                return compute_psd_from_realsht(sht(t_m))

            p_m1, p_m2 = get_member_psd(m1), get_member_psd(m2)

            sum_mean += p_mean; sum_m1 += p_m1; sum_m2 += p_m2; sum_truth += p_truth
            count += 1
        except Exception: continue

    # Calculate Ratios (Square root of the Power Ratio = Amplitude Ratio)
    degs = np.arange(p_truth.shape[-1])
    r_mean = torch.sqrt(sum_mean / sum_truth).cpu().numpy()
    r_m1 = torch.sqrt(sum_m1 / sum_truth).cpu().numpy()
    r_m2 = torch.sqrt(sum_m2 / sum_truth).cpu().numpy()

    # Plotting
    ax.axhline(1.0, linestyle=":", color="gray")
    ax.semilogx(degs[1:], r_mean[1:], label="Ens. Mean", lw=3, color="indigo")
    ax.semilogx(degs[1:], r_m1[1:], label=f"Mem {m1}", lw=1.5, alpha=0.6, color="tab:blue")
    ax.semilogx(degs[1:], r_m2[1:], label=f"Mem {m2}", lw=1.5, alpha=0.6, color="tab:red")

    ax.set_title(cfg["label"], fontweight="bold")
    ax.set_xlabel("Wavenumber")
    if ax == axes[0]: ax.set_ylabel("Amplitude Ratio")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, fontsize=12)

plt.tight_layout()
# -----------------------------
# Save Figure
# -----------------------------
output_filename = "spectral_amplitude_ratios.png"

# Use 300 DPI for high-quality publication-ready output
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Figure successfully saved as: {output_filename}")
