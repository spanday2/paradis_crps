"""CRPS utilities for ensemble training.

This file contains the two-member almost-fair CRPS loss and the
memory-efficient streaming training/validation routines used by LitParadis.
"""

import torch
import torch.nn as nn


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

        loss_x1_fit = litmodel.crps_loss.fit_term(x1, y) / num_steps
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
        loss_x2 = (fit_x2 - spread_x2) / num_steps

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
        loss_x1_spread = (-spread_x1) / num_steps

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

    return batch_loss, batch_fit, batch_spread


@torch.no_grad()
def two_member_afcrps_validation_step(
    litmodel,
    input_data: torch.Tensor,
    true_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-member afCRPS validation step.

    Returns:
        val_loss, val_fit, val_spread, report_loss
    """

    batch_size = input_data.size(0)
    num_steps = input_data.size(1)

    val_loss = input_data.new_zeros(())
    val_fit = input_data.new_zeros(())
    val_spread = input_data.new_zeros(())

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

        val_loss += loss_step
        val_fit += fit_step
        val_spread += spread_step

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

    return val_loss, val_fit, val_spread, report_loss