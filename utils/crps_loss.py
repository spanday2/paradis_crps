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
        Compute the full-order spectral absolute-coefficient difference.

        RealSHT stores only m >= 0. The omitted negative orders are
        included through multiplicity weights:

            m = 0   -> weight 1
            m > 0   -> weight 2
            m > ell -> weight 0

        The spectral dimensions L and M are summed. The result is averaged
        only over batch and variables, then divided by 4*pi before the
        external spectral loss weight is applied.
        """
        a_hat = self._sht_coeffs(a)
        b_hat = self._sht_coeffs(b)

        # Shape: [B, C, L, M]
        abs_diff = torch.abs(a_hat - b_hat)

        B, C, L, M = abs_diff.shape

        # Shape: [1, C, 1, 1]
        var_weights = self.var_loss_weights.to(
            device=abs_diff.device,
            dtype=abs_diff.dtype,
        ).view(1, C, 1, 1)

        # Shape: [L, M]
        #
        # m=0 is counted once, m>0 twice, and m>ell is excluded.
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

        # Sum over L and M.
        # Average only over B and C.
        full_spectral_afcrps_distance = (
            weighted_abs_diff
            .sum(dim=(-2, -1))
            .mean()
        )

        # Normalize before multiplication by spectral_crps_weight.
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
    
    

def two_member_afcrps_training_step(
    litmodel,
    input_data: torch.Tensor,
    true_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-efficient two-member afCRPS training step.

    This function does the ensemble part of LitParadis.training_step.

    It assumes:
        - litmodel.automatic_optimization = False
        - litmodel.num_members == 2
        - litmodel.crps_loss is a TwoMemberAlmostFairCRPS instance

    Returns:
        batch_loss, batch_fit, batch_spread
    """

    batch_size = input_data.size(0)
    num_steps = input_data.size(1)

    opt = litmodel.optimizers()
    opt.zero_grad(set_to_none=True)

    member1_inputs = input_data.clone()
    member2_inputs = input_data.clone()

    log_loss = input_data.new_zeros(())
    log_fit = input_data.new_zeros(())
    log_spread = input_data.new_zeros(())
    
    log_spectral_loss = input_data.new_zeros(())
    log_total_loss = input_data.new_zeros(())

    for step in range(num_steps):
        y = true_data[:, step]

        # Same raw_noise1 is reused for x1 and x1_re.
        # The embedding MLP is applied twice, creating fresh graphs.
        raw_noise1 = litmodel._sample_raw_noise(
            batch_size=batch_size,
            device=input_data.device,
            dtype=input_data.dtype,
        )
        raw_noise2 = litmodel._sample_raw_noise(
            batch_size=batch_size,
            device=input_data.device,
            dtype=input_data.dtype,
        )

        # --------------------------------------------------
        # 1) Member 1 fit gradient only
        # --------------------------------------------------
        noise1 = litmodel._embed_raw_noise(raw_noise1)
        x1 = litmodel.forward(member1_inputs[:, step], noise_emb=noise1)

        loss_x1_fit = litmodel.crps_loss.fit_term(x1, y)
        if litmodel.spectral_crps_loss is not None:
            loss_x1_fit = loss_x1_fit + litmodel.spectral_crps_weight * (
                litmodel.spectral_crps_loss.fit_term(x1, y)
            )

        loss_x1_fit = loss_x1_fit / num_steps
        litmodel.manual_backward(loss_x1_fit)

        x1_det = x1.detach()

        del x1
        del loss_x1_fit
        del noise1

        # --------------------------------------------------
        # 2) Member 2 fit + spread gradient w.r.t. x2
        # --------------------------------------------------
        noise2 = litmodel._embed_raw_noise(raw_noise2)
        x2 = litmodel.forward(member2_inputs[:, step], noise_emb=noise2)

        fit_x2 = litmodel.crps_loss.fit_term(x2, y)
        spread_x2 = litmodel.crps_loss.spread_term(x1_det, x2)

        loss_x2 = fit_x2 - spread_x2

        if litmodel.spectral_crps_loss is not None:
            spec_fit_x2 = litmodel.spectral_crps_loss.fit_term(x2, y)
            spec_spread_x2 = litmodel.spectral_crps_loss.spread_term(x1_det, x2)

            loss_x2 = loss_x2 + litmodel.spectral_crps_weight * (
                spec_fit_x2 - spec_spread_x2
            )

        loss_x2 = loss_x2 / num_steps

        litmodel.manual_backward(loss_x2)

        x2_det = x2.detach()

        del x2
        del fit_x2
        del spread_x2
        del loss_x2
        del noise2

        # --------------------------------------------------
        # 3) Recompute member 1 with same raw noise values,
        #    but with a fresh noise-embedding graph.
        # --------------------------------------------------
        noise1_re = litmodel._embed_raw_noise(raw_noise1)
        x1_re = litmodel.forward(member1_inputs[:, step], noise_emb=noise1_re)

        spread_x1 = litmodel.crps_loss.spread_term(x1_re, x2_det)
        loss_x1_spread = -spread_x1

        if litmodel.spectral_crps_loss is not None:
            spec_spread_x1 = litmodel.spectral_crps_loss.spread_term(x1_re, x2_det)
            loss_x1_spread = loss_x1_spread - litmodel.spectral_crps_weight * spec_spread_x1

        loss_x1_spread = loss_x1_spread / num_steps

        litmodel.manual_backward(loss_x1_spread)

        x1_re_det = x1_re.detach()

        del x1_re
        del spread_x1
        del loss_x1_spread
        del noise1_re

        # Logging only; no graph kept.
        with torch.no_grad():
            step_loss, step_fit, step_spread = litmodel.crps_loss.full_loss_for_logging(
                x1_re_det,
                x2_det,
                y,
            )

            log_loss += step_loss
            log_fit += step_fit
            log_spread += step_spread

            step_total_loss = step_loss

            if litmodel.spectral_crps_loss is not None:
                spec_loss, spec_fit, spec_spread = (
                    litmodel.spectral_crps_loss.full_loss_for_logging(
                        x1_re_det,
                        x2_det,
                        y,
                    )
                )

                weighted_spec_loss = litmodel.spectral_crps_weight * spec_loss

                log_spectral_loss += weighted_spec_loss

                step_total_loss = step_total_loss + weighted_spec_loss

            log_total_loss += step_total_loss

        # Autoregressive propagation uses detached states.
        member1_inputs = litmodel._autoregression_input_from_output(
            member1_inputs,
            x1_re_det,
            step,
            num_steps,
        )
        member2_inputs = litmodel._autoregression_input_from_output(
            member2_inputs,
            x2_det,
            step,
            num_steps,
        )

        del x1_det
        del x2_det
        del x1_re_det
        del raw_noise1
        del raw_noise2

    litmodel._optimizer_and_scheduler_step()

    batch_loss = log_loss / num_steps
    batch_fit = log_fit / num_steps
    batch_spread = log_spread / num_steps

    batch_spectral_loss = log_spectral_loss / num_steps

    batch_total_loss = log_total_loss / num_steps

    return (
        batch_loss,
        batch_fit,
        batch_spread,
        batch_spectral_loss,
        batch_total_loss,
    )


@torch.no_grad()
def two_member_afcrps_validation_step(
    litmodel,
    input_data: torch.Tensor,
    true_data: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Two-member afCRPS validation step.

    Returns:
        val_loss, val_fit, val_spread,
        val_spectral_loss, val_total_loss,
        report_loss
    """

    batch_size = input_data.size(0)
    num_steps = input_data.size(1)

    val_loss = input_data.new_zeros(())
    val_fit = input_data.new_zeros(())
    val_spread = input_data.new_zeros(())

    val_spectral_loss = input_data.new_zeros(())
    val_total_loss = input_data.new_zeros(())

    report_loss = torch.zeros(
        len(litmodel.report_ind),
        dtype=input_data.dtype,
        device=input_data.device,
    )

    member1_inputs = input_data.clone()
    member2_inputs = input_data.clone()

    for step in range(num_steps):
        y = true_data[:, step]

        raw_noise1 = litmodel._sample_raw_noise(
            batch_size=batch_size,
            device=input_data.device,
            dtype=input_data.dtype,
        )
        raw_noise2 = litmodel._sample_raw_noise(
            batch_size=batch_size,
            device=input_data.device,
            dtype=input_data.dtype,
        )

        noise1 = litmodel._embed_raw_noise(raw_noise1)
        noise2 = litmodel._embed_raw_noise(raw_noise2)

        x1 = litmodel.forward(member1_inputs[:, step], noise_emb=noise1)
        x2 = litmodel.forward(member2_inputs[:, step], noise_emb=noise2)

        loss_step, fit_step, spread_step = litmodel.crps_loss.full_loss_for_logging(
            x1,
            x2,
            y,
        )

        total_loss_step = loss_step

        if litmodel.spectral_crps_loss is not None:
            spec_loss_step, _, _ = litmodel.spectral_crps_loss.full_loss_for_logging(
                x1,
                x2,
                y,
            )

            weighted_spec_loss_step = (
                litmodel.spectral_crps_weight * spec_loss_step
            )

            val_spectral_loss += weighted_spec_loss_step
            total_loss_step = total_loss_step + weighted_spec_loss_step

        val_loss += loss_step
        val_fit += fit_step
        val_spread += spread_step
        val_total_loss += total_loss_step

        mean_out = 0.5 * (x1 + x2)
        report_loss += litmodel._get_report_rmse(mean_out, y)

        member1_inputs = litmodel._autoregression_input_from_output(
            member1_inputs,
            x1.detach(),
            step,
            num_steps,
        )
        member2_inputs = litmodel._autoregression_input_from_output(
            member2_inputs,
            x2.detach(),
            step,
            num_steps,
        )

        del raw_noise1
        del raw_noise2
        del noise1
        del noise2
        del x1
        del x2
        del mean_out

    val_loss = val_loss / num_steps
    val_fit = val_fit / num_steps
    val_spread = val_spread / num_steps
    val_spectral_loss = val_spectral_loss / num_steps
    val_total_loss = val_total_loss / num_steps

    return (
        val_loss,
        val_fit,
        val_spread,
        val_spectral_loss,
        val_total_loss,
        report_loss,
    )