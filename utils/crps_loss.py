"""CRPS utilities for ensemble training.

This file contains the two-member almost-fair CRPS loss and the
memory-efficient streaming training/validation routines used by LitParadis.
"""
import math
import torch
import torch.nn as nn
import torch_harmonics as th


class TwoMemberAlmostFairCRPS(nn.Module):
    """Two-member almost-fair CRPS with variable and latitude weighting.

    For two members, the loss is
        0.5 * |x1 - y| + 0.5 * |x2 - y| - 0.5 * C * |x1 - x2|
    """

    def __init__(
        self,
        var_loss_weights: torch.Tensor,
        lat_grid: torch.Tensor | None = None,
        alpha: float = 0.95,
        pairwise_coeff: float | None = None,
        apply_latitude_weights: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.pairwise_coeff = pairwise_coeff
        self.apply_latitude_weights = apply_latitude_weights

        # Register variable weights as buffer for device management
        self.register_buffer("var_loss_weights", var_loss_weights.float().detach())

        if apply_latitude_weights and lat_grid is not None:
            self.register_buffer("lat_weights", self._compute_latitude_weights(lat_grid).detach())
        else:
            self.lat_weights = None

    @property
    def c(self) -> float:
        if self.pairwise_coeff is not None:
            return float(self.pairwise_coeff)
        eps = (1.0 - float(self.alpha)) / 2.0
        return 1.0 - eps

    def _compute_latitude_weights(self, grid_lat_deg: torch.Tensor) -> torch.Tensor:
        """GraphCast-consistent latitude weights (unit-mean)."""
        lat = grid_lat_deg.to(dtype=torch.float64)
        if lat.ndim != 1:
            raise ValueError(f"grid_lat_deg must be 1D [H], got {lat.shape}")

        d = lat[1:] - lat[:-1]
        d0 = d[0]
        if not torch.allclose(d, d0.expand_as(d), rtol=0.0, atol=1e-6):
            raise ValueError("Latitude grid is not uniformly spaced.")

        delta = torch.abs(d0)
        lat_min = torch.min(lat)
        lat_max = torch.max(lat)

        has_poles = torch.isclose(
            lat_min, lat.new_tensor(-90.0), atol=1e-6
        ) and torch.isclose(lat_max, lat.new_tensor(90.0), atol=1e-6)

        if has_poles:
            lat_rad = torch.deg2rad(lat)
            delta_rad = torch.deg2rad(delta)
            weights = torch.cos(lat_rad) * torch.sin(delta_rad / 2.0)
            pole_w = torch.sin(delta_rad / 4.0) ** 2
            weights[torch.argmin(lat)] = pole_w
            weights[torch.argmax(lat)] = pole_w
        else:
            weights = torch.cos(torch.deg2rad(lat))

        weights = weights / weights.mean()
        return weights.to(dtype=grid_lat_deg.dtype)

    def _weighted_mean_abs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the weighted mean absolute difference across B, C, Lat, Lon."""
        # Shape: [B, C, Lat, Lon]
        abs_diff = torch.abs(a - b)
        
        # Apply variable weights: broadcast to [1, C, 1, 1]
        var_weights = self.var_loss_weights.view(1, -1, 1, 1)
        abs_diff = abs_diff * var_weights

        # Apply latitude weights: broadcast to [1, 1, Lat, 1]
        if self.apply_latitude_weights and self.lat_weights is not None:
            lat_weights = self.lat_weights.view(1, 1, -1, 1)
            abs_diff = abs_diff * lat_weights

        return torch.mean(abs_diff)

    def fit_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * self._weighted_mean_abs(x, y)

    def spread_term(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.c * self._weighted_mean_abs(x1, x2)

    def full_loss_for_logging(
        self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fit = self.fit_term(x1, y) + self.fit_term(x2, y)
        spread = self.spread_term(x1, x2)
        loss = fit - spread
        return loss, fit, spread


class TwoMemberSpectralAlmostFairCRPS(nn.Module):
    """Two-member almost-fair afCRPS in spectral space with variable weighting.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        var_loss_weights: torch.Tensor,
        alpha: float = 0.95,
        pairwise_coeff: float | None = None,
        grid: str = "equiangular",
    ):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.alpha = alpha
        self.pairwise_coeff = pairwise_coeff

        self.register_buffer("var_loss_weights", var_loss_weights.float().detach())
        self.sht = th.RealSHT(nlat, nlon, grid=grid, norm="ortho")

    @property
    def c(self) -> float:
        if self.pairwise_coeff is not None:
            return float(self.pairwise_coeff)
        eps = (1.0 - float(self.alpha)) / 2.0
        return 1.0 - eps

    def _sht_coeffs(self, x: torch.Tensor) -> torch.Tensor:
        B, C, lat, lon = x.shape
        if lat != self.nlat or lon != self.nlon:
            raise ValueError(f"Expected spatial shape ({self.nlat}, {self.nlon}), got ({lat}, {lon}).")

        x_2d = x.float().reshape(B * C, lat, lon)
        coeffs = self.sht(x_2d)
        # FCN3-style coefficient normalization.
        coeffs = coeffs / math.sqrt(4.0 * math.pi)
        return coeffs.reshape(B, C, coeffs.shape[-2], coeffs.shape[-1])
    
    def _make_real_sht_mode_weights(
        self,
        L: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Construct full-order weights for RealSHT coefficients.

        RealSHT stores only nonnegative orders m >= 0.

            m = 0   -> weight 1
            m > 0   -> weight 2
            m > ell -> weight 0
        """
        ell = torch.arange(
            L,
            device=device,
        ).view(L, 1)

        m = torch.arange(
            M,
            device=device,
        ).view(1, M)

        valid = m <= ell

        mode_weights = torch.ones(
            (L, M),
            device=device,
            dtype=dtype,
        )

        if M > 1:
            mode_weights[:, 1:] = 2.0

        mode_weights = mode_weights * valid.to(dtype)

        return mode_weights

    def _weighted_mean_abs_coeff(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the full-order absolute difference between spectral
        coefficient magnitudes.

        For each spherical-harmonic mode, this computes:

            abs(abs(SHT(a)) - abs(SHT(b)))

        rather than:

            abs(SHT(a) - SHT(b))

        Therefore, the loss compares spectral amplitudes and ignores
        spectral phase differences.

        RealSHT stores only m >= 0. The omitted negative orders are
        included through multiplicity weights:

            m = 0   -> weight 1
            m > 0   -> weight 2
            m > ell -> weight 0

        The spectral dimensions L and M are summed. The result is averaged
        over batch and variables, then divided by 4*pi before the external
        spectral loss weight is applied.
        """
        a_hat = self._sht_coeffs(a)
        b_hat = self._sht_coeffs(b)

        # Convert complex SHT coefficients to non-negative amplitudes.
        a_amplitude = torch.abs(a_hat)
        b_amplitude = torch.abs(b_hat)

        # Compare spectral amplitudes:
        #
        #     ||a_hat| - |b_hat||
        #
        # instead of:
        #
        #     |a_hat - b_hat|
        abs_diff = torch.abs(
            a_hat - b_hat
        )

        B, C, L, M = abs_diff.shape

        # Variable weights: [1, C, 1, 1]
        var_weights = self.var_loss_weights.to(
            device=abs_diff.device,
            dtype=abs_diff.dtype,
        ).view(1, C, 1, 1)

        # RealSHT multiplicity weights: [L, M]
        #
        # m = 0 is counted once.
        # m > 0 is counted twice.
        # m > ell is excluded.
        mode_weights = self._make_real_sht_mode_weights(
            L=L,
            M=M,
            device=abs_diff.device,
            dtype=abs_diff.dtype,
        )

        weighted_abs_diff = (
            abs_diff
            * var_weights
            * mode_weights.view(1, 1, L, M)
        )

        # Sum over spectral modes.
        # Average over batch and variables.
        full_spectral_afcrps_distance = (
            weighted_abs_diff
            .sum(dim=(-2, -1))
            .mean()
        )

        # Apply one spherical normalization factor.
        four_pi = full_spectral_afcrps_distance.new_tensor(
            4.0 * math.pi
        )

        return full_spectral_afcrps_distance / four_pi


    def fit_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * self._weighted_mean_abs_coeff(x, y)

    def spread_term(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.c * self._weighted_mean_abs_coeff(x1, x2)

    def full_loss_for_logging(
        self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fit = self.fit_term(x1, y) + self.fit_term(x2, y)
        spread = self.spread_term(x1, x2)
        loss = fit - spread
        return loss, fit, spread