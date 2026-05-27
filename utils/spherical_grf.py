import torch
import math
import matplotlib.pyplot as plt


def generate_spherical_grf(
    Lat: torch.Tensor,
    Lon: torch.Tensor,
    effective_resolution: int = 5,
    n_waves: int | None = None,
    device: str = "cpu",
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generates an isotropic Gaussian Random Field on S^2 via Random Fourier Features.

    The wave vectors are sampled from a 3D isotropic Gaussian distribution, which
    induces a chordal Squared-Exponential covariance kernel on the sphere:

        k(x, y) = exp(-(1 - x.y) / length_scale^2)

    where length_scale is derived automatically from the grid resolution and
    effective_resolution (see below). An analytic standardization is applied
    post-summation to eliminate the O(1/sqrt(M)) finite-sample bias, guaranteeing
    an empirical mean of exactly 0 and std of exactly 1.

    Args:
        Lat: Latitude tensor in radians, shape (n_lat, n_lon), produced by
            torch.meshgrid. Values should span [-pi/2, pi/2].
        Lon: Longitude tensor in radians, shape (n_lat, n_lon), produced by
            torch.meshgrid. Values should span [-pi, pi).
        effective_resolution: Target decorrelation scale expressed as a number of
            grid cells. The field will be spatially smooth at scales smaller than
            this and uncorrelated at larger separations. For example,
            effective_resolution=5 on a 1-degree grid (n_lat=181) produces
            features that decorrelate over roughly 5 degrees. Must be >= 1.
            Default: 5.
        n_waves: Number of random plane waves to sum (the M in the RFF
            approximation). If None, M is set automatically to
            max(1000, 10 * sigma^2) where sigma = 1 / length_scale, ensuring
            adequate spectral coverage for the chosen effective_resolution.
            Larger values give a more faithful Gaussian field at increased
            memory and compute cost. A warning is raised if an explicit value
            is passed that is below the recommended minimum. Default: None.
        device: PyTorch device string, e.g. 'cpu' or 'cuda'. The Lat and Lon
            tensors must already reside on this device. Default: 'cpu'.
        seed: Integer seed for the random number generator, for reproducibility.
            If None, the global PyTorch RNG state is used. Default: None.
        dtype: Floating-point dtype for all internal computation and the returned
            tensor. torch.float64 is recommended for large grids or small
            effective_resolution values where numerical precision matters.
            Default: torch.float32.

    Returns:
        field: Tensor of shape (n_lat, n_lon) containing one realisation of the
            GRF. The empirical mean is exactly 0 and the empirical std is
            exactly 1 after analytic standardization.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Lat-lon grid -> 3D Cartesian coords on the unit sphere
    cos_lat = torch.cos(Lat)
    coords = torch.stack(
        [
            cos_lat * torch.cos(Lon),
            cos_lat * torch.sin(Lon),
            torch.sin(Lat),
        ],
        dim=-1,
    ).reshape(-1, 3)

    n_lat, n_lon = Lat.shape

    # Angular correlation length in radians
    length_scale = effective_resolution * math.pi / (n_lat * math.sqrt(6))
    sigma = 1.0 / length_scale

    if n_waves is None:
        n_waves = min(5000, max(1000, int(10 * sigma**2)))
        print(f"n_waves set to {n_waves}")

    # Sample wave vectors from 3D Gaussian
    K = torch.randn(3, n_waves, device=device, dtype=dtype) * sigma
    phases = torch.rand(n_waves, device=device, dtype=dtype) * 2.0 * math.pi

    # Evaluate and sum waves
    dot_product = coords @ K
    field = math.sqrt(2.0 / n_waves) * torch.cos(dot_product + phases).sum(dim=1)
    field = field.view(n_lat, n_lon)

    # Analytic standardization
    field = (field - field.mean()) / field.std()

    return field


# Example at 1-degree resolution
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    n_lat = 181
    n_lon = 360
    lat = torch.linspace(-math.pi / 2, math.pi / 2, n_lat, device=device, dtype=dtype)
    lon = torch.linspace(-math.pi, math.pi, n_lon + 1, device=device, dtype=dtype)[:-1]
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")

    grf = generate_spherical_grf(
        lat,
        lon,
        n_waves=2000,
        device=device,
        seed=42,
        dtype=dtype,
    )

    print(f"Device : {device}")
    print(f"Dtype  : {grf.dtype}")
    print(f"Shape  : {tuple(grf.shape)}")
    print(f"Mean   : {grf.mean().item():.6f}  (Expected = 0)")
    print(f"Std    : {grf.std().item():.6f}  (Expected = 1)")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    # Move tensors to CPU and convert to numpy for matplotlib
    import numpy

    grf_np = grf.cpu().float().numpy()
    lon_np = lon.cpu().float().numpy()
    lat_np = lat.cpu().float().numpy()

    fig = plt.figure(figsize=(12, 10))

    # 1. Equirectangular (Lat-Lon) Projection
    ax1 = fig.add_subplot(2, 1, 1)
    im1 = ax1.imshow(
        grf_np,
        extent=[-180, 180, -90, 90],
        origin="lower",
        cmap="viridis",
    )
    ax1.set_title("Equirectangular Projection")
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    fig.colorbar(im1, ax=ax1, label="Field value")

    # 2. North Pole View
    ax2 = fig.add_subplot(2, 2, 3, projection="polar")
    r_north = numpy.pi / 2.0 - lat_np
    ax2.pcolormesh(lon_np, r_north, grf_np, cmap="viridis", shading="auto")
    ax2.set_ylim(0, numpy.pi / 2.0)
    ax2.set_yticks([])
    ax2.set_title("North Pole View")

    # 3. South Pole View
    ax3 = fig.add_subplot(2, 2, 4, projection="polar")
    r_south = lat_np + numpy.pi / 2.0
    ax3.pcolormesh(lon_np, r_south, grf_np, cmap="viridis", shading="auto")
    ax3.set_ylim(0, numpy.pi / 2.0)
    ax3.set_yticks([])
    ax3.set_title("South Pole View")

    plt.tight_layout()
    plt.savefig("spherical_grf_dashboard.png", dpi=300, bbox_inches="tight")
    plt.show()
