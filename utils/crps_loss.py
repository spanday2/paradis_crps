"""CRPS utilities for ensemble training.

This file contains the two-member almost-fair CRPS loss and the
memory-efficient streaming training/validation routines used by LitParadis.
"""

import torch
import torch.nn as nn
import torch_harmonics as th


class TwoMemberAlmostFairCRPS(nn.Module):
    """Two-member almost-fair CRPS.

    For two members, the loss is

        0.5 * |x1 - y| + 0.5 * |x2 - y| - 0.5 * C * |x1 - x2|

    where

        eps = (1 - alpha) / 2
        C   = 1 - eps

    If alpha=1, this gives the fair two-member coefficient C=1.

    If the standard unfair CRPS coefficient for M=2 is desired, set
    pairwise_coeff=0.5.
    """

    def __init__(self, alpha: float = 0.95, pairwise_coeff: float | None = None):
        super().__init__()
        self.alpha = alpha
        self.pairwise_coeff = pairwise_coeff

    @property
    def c(self) -> float:
        if self.pairwise_coeff is not None:
            return float(self.pairwise_coeff)

        eps = (1.0 - float(self.alpha)) / 2.0
        return 1.0 - eps

    @staticmethod
    def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(a - b))

    def fit_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * self._mean_abs(x, y)

    def spread_term(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.c * self._mean_abs(x1, x2)

    def full_loss_for_logging(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fit = self.fit_term(x1, y) + self.fit_term(x2, y)
        spread = self.spread_term(x1, x2)
        loss = fit - spread
        return loss, fit, spread
    
class TwoMemberSpectralAlmostFairCRPS(nn.Module):
    """Two-member almost-fair afCRPS in spectral space.

    This computes afCRPS after applying spherical harmonic transform
    channel-by-channel.

    It computes afCRPS per channel and spectral mode, then averages.

    For two members:

        0.5 * |x1_hat - y_hat|
      + 0.5 * |x2_hat - y_hat|
      - 0.5 * C * |x1_hat - x2_hat|

    where _hat indicates complex SHT coefficients.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        alpha: float = 0.95,
        pairwise_coeff: float | None = None,
        grid: str = "equiangular",
    ):
        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.alpha = alpha
        self.pairwise_coeff = pairwise_coeff

        self.sht = th.RealSHT(nlat, nlon, grid=grid)

    @property
    def c(self) -> float:
        if self.pairwise_coeff is not None:
            return float(self.pairwise_coeff)

        eps = (1.0 - float(self.alpha)) / 2.0
        return 1.0 - eps

    def _sht_coeffs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RealSHT to each channel independently.

        Input:
            x: B, C, lat, lon

        Output:
            coeffs: B, C, L, M
        """
        B, C, lat, lon = x.shape

        if lat != self.nlat or lon != self.nlon:
            raise ValueError(
                f"Expected spatial shape ({self.nlat}, {self.nlon}), "
                f"got ({lat}, {lon})."
            )

        # RealSHT is safer in float32 under AMP/bfloat16.
        x_2d = x.float().reshape(B * C, lat, lon)

        coeffs = self.sht(x_2d)

        coeffs = coeffs.reshape(B, C, coeffs.shape[-2], coeffs.shape[-1])

        return coeffs

    def _mean_abs_coeff(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mean absolute difference of complex spectral coefficients."""
        a_hat = self._sht_coeffs(a)
        b_hat = self._sht_coeffs(b)

        return torch.mean(torch.abs(a_hat - b_hat))

    def fit_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * self._mean_abs_coeff(x, y)

    def spread_term(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.c * self._mean_abs_coeff(x1, x2)

    def full_loss_for_logging(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y: torch.Tensor,
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