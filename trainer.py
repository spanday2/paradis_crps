"""Model training implementation — ensemble CRPS version."""

import datetime
import logging
import re
import time
from collections import defaultdict

import lightning as L
import omegaconf.dictconfig
import torch
import torch.nn as nn
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_only

from data.datamodule import Era5DataModule
from model.paradis import Paradis
from utils.loss import ParadisLoss
from utils.normalization import denormalize_humidity, denormalize_precipitation


def _allreduce_scalar(x: torch.Tensor, op: str):
    if not (dist.is_available() and dist.is_initialized()):
        return x

    y = x.detach().clone()

    if op == "max":
        dist.all_reduce(y, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(y, op=dist.ReduceOp.MIN)
    else:
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= dist.get_world_size()

    return y


class TwoMemberAlmostFairCRPS(nn.Module):
    """Memory-efficient two-member almost-fair CRPS utilities.

    For two members, the loss is

        0.5 * |x1 - y| + 0.5 * |x2 - y| - 0.5 * C * |x1 - x2|

    where, matching the previous AlmostFairCRPS implementation,

        eps = (1 - alpha) / 2
        C   = 1 - eps

    If alpha=1, this becomes the fair two-member coefficient C=1.
    If the standard unfair CRPS coefficient for M=2 is desired, set
    training.crps_pairwise_coeff = 0.5 in the config.
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


class LitParadis(L.LightningModule):
    """Lightning module for Paradis ensemble.

    Ensemble mode is activated by setting cfg.model.noise_channels > 0.
    When noise_channels == 0 the module is identical to deterministic Paradis.
    """

    model: torch.nn.Module

    def __init__(
        self,
        datamodule: Era5DataModule,
        cfg: omegaconf.dictconfig.DictConfig,
    ) -> None:
        super().__init__()

        self.min_dt = 1e10
        self.datamodule = datamodule

        lat_grid = datamodule.dataset.lat_rad_grid
        lon_grid = datamodule.dataset.lon_rad_grid

        self.model = Paradis(datamodule, cfg, lat_grid, lon_grid)
        self.cfg = cfg
        self.n_inputs = cfg.dataset.n_time_inputs

        # ------------------------------------------------------------------ #
        # Ensemble configuration
        # ------------------------------------------------------------------ #
        self.noise_channels = cfg.model.get("noise_channels", 0)
        self.ensemble_mode = self.noise_channels > 0
        self.num_members = cfg.training.get("num_ensemble_members", 2)

        self.nlat = lat_grid.shape[0]
        self.nlon = lat_grid.shape[1]

        # if self.ensemble_mode and self.num_members != 2:
        #     raise ValueError(
        #         "The memory-efficient streaming afCRPS trainer is implemented only "
        #         f"for two members, but got num_ensemble_members={self.num_members}."
        #     )

        # if self.ensemble_mode:
        #     self.automatic_optimization = False
        if self.ensemble_mode and not cfg.forecast.enable:
            if self.num_members != 2:
                raise ValueError(
                    "The memory-efficient streaming afCRPS trainer is implemented only for "
                    f"two training members, but got num_ensemble_members={self.num_members}."
                )

            self.automatic_optimization = False

        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        if self.global_rank == 0:
            logging.info("Number of trainable parameters: {:,}".format(num_parameters))

            if self.ensemble_mode:
                logging.info(
                    f"Ensemble mode: {self.num_members} members, "
                    f"noise_channels={self.noise_channels}"
                )

        self.output_name_order = datamodule.output_name_order
        num_levels = len(cfg.features.pressure_levels)

        # ------------------------------------------------------------------ #
        # Variable loss weights
        # ------------------------------------------------------------------ #
        atmospheric_weights = torch.tensor(
            [
                cfg.training.variable_loss_weights.atmospheric[v]
                for v in cfg.features.output.atmospheric
            ],
            dtype=torch.float32,
        )

        surface_weights = torch.tensor(
            [
                cfg.training.variable_loss_weights.surface[v]
                for v in cfg.features.output.surface
            ],
            dtype=torch.float32,
        )

        atmospheric_vars = cfg.features.output.atmospheric
        surface_vars = cfg.features.output.surface

        var_name_to_weight = {
            **{v: atmospheric_weights[i] for i, v in enumerate(atmospheric_vars)},
            **{v: surface_weights[i] for i, v in enumerate(surface_vars)},
        }

        num_features = len(atmospheric_weights) * num_levels + len(surface_weights)
        var_loss_weights_reordered = torch.zeros(num_features, dtype=torch.float32)

        for i, var in enumerate(self.output_name_order):
            var_name = re.sub(r"_h\d+$", "", var)

            if var_name in var_name_to_weight:
                var_loss_weights_reordered[i] = var_name_to_weight[var_name]

        # ------------------------------------------------------------------ #
        # Loss functions
        # ------------------------------------------------------------------ #
        if self.ensemble_mode:
            alpha = cfg.training.get("crps_alpha", 0.95)
            pairwise_coeff = cfg.training.get("crps_pairwise_coeff", None)

            self.crps_loss = TwoMemberAlmostFairCRPS(
                alpha=alpha,
                pairwise_coeff=pairwise_coeff,
            )

            # Keep ParadisLoss for validation/report RMSE.
            self.loss_fn = ParadisLoss(
                loss_function="mse",
                lat_grid=datamodule.lat,
                pressure_levels=torch.tensor(
                    cfg.features.pressure_levels,
                    dtype=torch.float32,
                ),
                num_features=datamodule.num_out_features,
                num_surface_vars=len(cfg.features.output.surface),
                var_loss_weights=var_loss_weights_reordered,
                output_name_order=datamodule.output_name_order,
                delta_loss=cfg.training.loss_function.delta_loss,
                apply_latitude_weights=cfg.training.loss_function.lat_weights,
            )
        else:
            self.crps_loss = None

            self.loss_fn = ParadisLoss(
                loss_function=cfg.training.loss_function.type,
                lat_grid=datamodule.lat,
                pressure_levels=torch.tensor(
                    cfg.features.pressure_levels,
                    dtype=torch.float32,
                ),
                num_features=datamodule.num_out_features,
                num_surface_vars=len(cfg.features.output.surface),
                var_loss_weights=var_loss_weights_reordered,
                output_name_order=datamodule.output_name_order,
                delta_loss=cfg.training.loss_function.delta_loss,
                apply_latitude_weights=cfg.training.loss_function.lat_weights,
            )

        validation_loss_type = cfg.training.loss_function.get("validation_loss", None)

        if validation_loss_type is not None:
            self.val_loss_fn = ParadisLoss(
                loss_function=validation_loss_type,
                lat_grid=datamodule.lat,
                pressure_levels=torch.tensor(
                    cfg.features.pressure_levels,
                    dtype=torch.float32,
                ),
                num_features=datamodule.num_out_features,
                num_surface_vars=len(cfg.features.output.surface),
                var_loss_weights=var_loss_weights_reordered,
                output_name_order=datamodule.output_name_order,
                delta_loss=cfg.training.loss_function.delta_loss,
                apply_latitude_weights=cfg.training.loss_function.lat_weights,
            )
        else:
            self.val_loss_fn = self.loss_fn

        self.num_common_features = datamodule.num_common_features
        self.print_losses = cfg.training.print_losses

        if cfg.compute.compile:
            self.model.compile(
                mode="default",
                fullgraph=True,
                dynamic=False,
                backend="inductor",
            )

        if (cfg.init.checkpoint_path and not cfg.init.restart) or cfg.forecast.enable:
            checkpoint = torch.load(
                cfg.init.checkpoint_path,
                weights_only=True,
                map_location="cpu",
            )
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            self.load_state_dict(state_dict, strict=True)

        if not cfg.forecast.enable and cfg.training.reports.enable:
            self.report_features = cfg.training.reports.features
            self.report_ind = torch.tensor(
                [
                    datamodule.dataset.dyn_input_features.index(f)
                    for f in cfg.training.reports.features
                ],
                dtype=torch.long,
            )
            self.report_mean = torch.from_numpy(datamodule.dataset.report_stats["mean"])
            self.report_std = torch.from_numpy(datamodule.dataset.report_stats["std"])

        self.custom_norms = not cfg.normalization.standard

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _autoregression_input_from_output(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor,
        step: int,
        num_steps: int,
    ) -> torch.Tensor:
        new_input_data = input_data.clone()
        steps_left = num_steps - step - 1

        for i in range(min(steps_left, self.n_inputs)):
            beg_i = self.num_common_features * (self.n_inputs - i - 1)
            end_i = self.num_common_features * (self.n_inputs - i)

            new_input_data[:, step + i + 1, beg_i:end_i] = output_data[
                :, : self.num_common_features
            ]

        return new_input_data

    def _get_report_rmse(self, output_data, pred_data):
        lat_weights = self.loss_fn.lat_weights.view(1, 1, -1, 1).to(
            output_data.device
        )

        errors = torch.empty(
            len(self.report_ind),
            dtype=output_data.dtype,
            device=output_data.device,
        )

        for i, ind in enumerate(self.report_ind):
            if self.custom_norms and "specific_humidity" in self.report_features[i]:
                q_min = self.datamodule.dataset.q_min
                q_max = self.datamodule.dataset.q_max

                o_data = denormalize_humidity(output_data[:, ind], q_min, q_max)
                p_data = denormalize_humidity(pred_data[:, ind], q_min, q_max)

                errors[i] = torch.mean((o_data - p_data) ** 2 * lat_weights)

            elif self.custom_norms and "precipitation" in self.report_features[i]:
                o_data = denormalize_precipitation(output_data[:, ind])
                p_data = denormalize_precipitation(pred_data[:, ind])

                errors[i] = torch.mean((o_data - p_data) ** 2 * lat_weights)

            else:
                errors[i] = torch.mean(
                    ((output_data[:, ind] - pred_data[:, ind]) * self.report_std[i])
                    ** 2
                    * lat_weights
                )

        return torch.sqrt(errors).detach()

    def _sample_raw_noise(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample raw per-grid-point noise.

        Shape:
            (B, noise_channels, lat, lon)
        """
        return torch.randn(
            batch_size,
            self.noise_channels,
            self.nlat,
            self.nlon,
            device=device,
            dtype=dtype,
        )

    def _embed_raw_noise(self, raw_noise: torch.Tensor) -> torch.Tensor:
        """Pass raw spatial noise through the trainable noise embedding module.

        The NoiseEmbedding.forward() expects flattened noise of shape
        (N, noise_channels), so this helper reshapes the spatial field before
        and after applying the MLP.

        Args:
            raw_noise: (B, noise_channels, lat, lon)

        Returns:
            noise_emb: (B, emb_dim, lat, lon)
        """
        B, C_n, lat, lon = raw_noise.shape

        noise_flat = raw_noise.permute(0, 2, 3, 1).reshape(B * lat * lon, C_n)
        emb_flat = self.model.noise_embedding(noise_flat)

        emb_dim = emb_flat.shape[-1]
        noise_emb = emb_flat.reshape(B, lat, lon, emb_dim).permute(0, 3, 1, 2)

        return noise_emb

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        noise_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(x, noise_emb=noise_emb)

    # ---------------------------------------------------------------------- #
    # Training / validation
    # ---------------------------------------------------------------------- #

    def _optimizer_and_scheduler_step(self) -> None:
        """Manual optimisation step for ensemble mode."""
        opt = self.optimizers()
        opt.step()
        opt.zero_grad(set_to_none=True)

        sch = self.lr_schedulers()

        if sch is not None:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass
            else:
                sch.step()

    def training_step(self, batch, batch_idx):
        input_data, true_data = batch
        batch_size = input_data.size(0)
        num_steps = input_data.size(1)

        # -------------------------------------------------------------- #
        # Deterministic mode
        # -------------------------------------------------------------- #
        if not self.ensemble_mode:
            train_loss = 0.0

            for step in range(num_steps):
                output_data = self.forward(input_data[:, step])
                train_loss += self.loss_fn(output_data, true_data[:, step])

                input_data = self._autoregression_input_from_output(
                    input_data,
                    output_data,
                    step,
                    num_steps,
                )

            batch_loss = train_loss / num_steps
            self._last_train_loss_value = float(batch_loss.detach().item())

            self.log(
                "train_loss",
                batch_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                prog_bar=True,
            )
            self.log(
                "forecast_steps",
                num_steps,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )

            return batch_loss

        # -------------------------------------------------------------- #
        # Ensemble mode: memory-efficient two-member afCRPS.
        # -------------------------------------------------------------- #
        opt = self.optimizers()
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
            raw_noise1 = self._sample_raw_noise(
                batch_size=batch_size,
                device=input_data.device,
                dtype=input_data.dtype,
            )
            raw_noise2 = self._sample_raw_noise(
                batch_size=batch_size,
                device=input_data.device,
                dtype=input_data.dtype,
            )

            # ------------------------------
            # 1) Member 1 fit gradient only
            # ------------------------------
            noise1 = self._embed_raw_noise(raw_noise1)
            x1 = self.forward(member1_inputs[:, step], noise_emb=noise1)

            loss_x1_fit = self.crps_loss.fit_term(x1, y) / num_steps
            self.manual_backward(loss_x1_fit)

            x1_det = x1.detach()

            del x1
            del loss_x1_fit
            del noise1

            # ---------------------------------------------
            # 2) Member 2 fit + spread gradient w.r.t. x2
            # ---------------------------------------------
            noise2 = self._embed_raw_noise(raw_noise2)
            x2 = self.forward(member2_inputs[:, step], noise_emb=noise2)

            fit_x2 = self.crps_loss.fit_term(x2, y)
            spread_x2 = self.crps_loss.spread_term(x1_det, x2)
            loss_x2 = (fit_x2 - spread_x2) / num_steps

            self.manual_backward(loss_x2)

            x2_det = x2.detach()

            del x2
            del fit_x2
            del spread_x2
            del loss_x2
            del noise2

            # ------------------------------------------------------ #
            # 3) Recompute member 1 using the same raw noise values,
            #    but with a fresh noise-embedding graph.
            # ------------------------------------------------------ #
            noise1_re = self._embed_raw_noise(raw_noise1)
            x1_re = self.forward(member1_inputs[:, step], noise_emb=noise1_re)

            spread_x1 = self.crps_loss.spread_term(x1_re, x2_det)
            loss_x1_spread = (-spread_x1) / num_steps

            self.manual_backward(loss_x1_spread)

            x1_re_det = x1_re.detach()

            del x1_re
            del spread_x1
            del loss_x1_spread
            del noise1_re

            # Logging only; no graph kept.
            with torch.no_grad():
                step_loss, step_fit, step_spread = self.crps_loss.full_loss_for_logging(
                    x1_re_det,
                    x2_det,
                    y,
                )
                log_loss += step_loss
                log_fit += step_fit
                log_spread += step_spread

            # Autoregressive propagation uses detached states.
            member1_inputs = self._autoregression_input_from_output(
                member1_inputs,
                x1_re_det,
                step,
                num_steps,
            )
            member2_inputs = self._autoregression_input_from_output(
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

        self._optimizer_and_scheduler_step()

        batch_loss = log_loss / num_steps
        batch_fit = log_fit / num_steps
        batch_spread = log_spread / num_steps

        self._last_train_loss_value = float(batch_loss.detach().item())

        self.log(
            "train_loss",
            batch_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_crps_fit",
            batch_fit,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_crps_spread",
            batch_spread,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        self.log("lr", opt.param_groups[0]["lr"], prog_bar=True)
        self.log(
            "forecast_steps",
            num_steps,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        return batch_loss.detach()

    def validation_step(self, batch, batch_idx):
        input_data, true_data = batch
        batch_size = input_data.size(0)
        num_steps = input_data.size(1)

        val_loss = 0.0
        val_fit = 0.0
        val_spread = 0.0
        report_loss = 0.0

        if self.ensemble_mode:
            member1_inputs = input_data.clone()
            member2_inputs = input_data.clone()

            for step in range(num_steps):
                y = true_data[:, step]

                raw_noise1 = self._sample_raw_noise(
                    batch_size=batch_size,
                    device=input_data.device,
                    dtype=input_data.dtype,
                )
                raw_noise2 = self._sample_raw_noise(
                    batch_size=batch_size,
                    device=input_data.device,
                    dtype=input_data.dtype,
                )

                noise1 = self._embed_raw_noise(raw_noise1)
                noise2 = self._embed_raw_noise(raw_noise2)

                x1 = self.forward(member1_inputs[:, step], noise_emb=noise1)
                x2 = self.forward(member2_inputs[:, step], noise_emb=noise2)

                loss_step, fit_step, spread_step = self.crps_loss.full_loss_for_logging(
                    x1,
                    x2,
                    y,
                )

                val_loss += loss_step
                val_fit += fit_step
                val_spread += spread_step

                mean_out = 0.5 * (x1 + x2)
                report_loss += self._get_report_rmse(mean_out, y)

                member1_inputs = self._autoregression_input_from_output(
                    member1_inputs,
                    x1.detach(),
                    step,
                    num_steps,
                )
                member2_inputs = self._autoregression_input_from_output(
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

        else:
            for step in range(num_steps):
                output_data = self.forward(input_data[:, step])

                val_loss += self.val_loss_fn(output_data, true_data[:, step])
                report_loss += self._get_report_rmse(output_data, true_data[:, step])

                input_data = self._autoregression_input_from_output(
                    input_data,
                    output_data,
                    step,
                    num_steps,
                )

        self.log(
            "val_loss",
            val_loss / num_steps,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if self.ensemble_mode:
            self.log(
                "val_crps_fit",
                val_fit / num_steps,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "val_crps_spread",
                val_spread / num_steps,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        for i, name in enumerate(self.cfg.training.reports.features):
            self.log(
                name,
                report_loss[i] / num_steps,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return val_loss / num_steps

    # ---------------------------------------------------------------------- #
    # Optimiser / scheduler
    # ---------------------------------------------------------------------- #

    def configure_optimizers(self):
        cfg = self.cfg.training

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

        enabled_schedulers = sum(
            [
                cfg.scheduler.one_cycle.enabled,
                cfg.scheduler.reduce_lr.enabled,
                cfg.scheduler.wsd.enabled,
            ]
        )

        if enabled_schedulers != 1:
            raise ValueError(
                f"Exactly one scheduler must be enabled, found {enabled_schedulers}."
            )

        if cfg.scheduler.one_cycle.enabled:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                total_steps=int(self.trainer.estimated_stepping_batches),
                max_lr=cfg.optimizer.lr,
                pct_start=cfg.scheduler.one_cycle.warmup_pct_start,
                div_factor=cfg.scheduler.one_cycle.lr_div_factor,
                final_div_factor=cfg.scheduler.one_cycle.lr_final_div,
                anneal_strategy="cos",
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        if cfg.scheduler.reduce_lr.enabled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.scheduler.reduce_lr.factor,
                patience=cfg.scheduler.reduce_lr.patience,
                threshold=cfg.scheduler.reduce_lr.threshold,
                threshold_mode=cfg.scheduler.reduce_lr.threshold_mode,
                min_lr=cfg.scheduler.reduce_lr.min_lr,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if cfg.scheduler.wsd.enabled:
            total_steps = self.trainer.estimated_stepping_batches

            warmup_steps = (
                cfg.scheduler.wsd.warmup
                if cfg.scheduler.wsd.warmup >= 1
                else cfg.scheduler.wsd.warmup * total_steps
            )
            decay_steps = (
                cfg.scheduler.wsd.decay
                if cfg.scheduler.wsd.decay >= 1
                else cfg.scheduler.wsd.decay * total_steps
            )

            assert warmup_steps >= 0 and decay_steps >= 0
            assert warmup_steps + decay_steps <= total_steps

            steady_steps = total_steps - (warmup_steps + decay_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                if step <= warmup_steps + steady_steps:
                    return 1.0
                return (total_steps - step) / decay_steps

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        raise RuntimeError("No scheduler was configured.")

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #

    @rank_zero_only
    def on_fit_start(self):
        total = sum(p.numel() for p in self.parameters())

        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_scalar(
                "model/num_parameters",
                total,
                global_step=0,
            )

    def on_train_epoch_start(self):
        if self.print_losses:
            self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.print_losses and self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_loss = self.trainer.callback_metrics.get("train_loss")
            val_loss = self.trainer.callback_metrics.get("val_loss")

            if (
                self.trainer.is_global_zero
                and train_loss is not None
                and val_loss is not None
            ):
                print(
                    f"Epoch {self.current_epoch:4d} | "
                    f"Train Loss: {train_loss.item():.6f} | "
                    f"Val Loss: {val_loss.item():.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Elapsed time: {elapsed_time:.4f}s"
                )

    def on_train_end(self):
        logging.info(f"Training completed after {self.current_epoch + 1} epochs")

    def on_before_optimizer_step(self, optimizer):
        grad_sq = defaultdict(lambda: torch.zeros((), device=self.device))
        param_sq = defaultdict(lambda: torch.zeros((), device=self.device))
        momentum_sq = defaultdict(lambda: torch.zeros((), device=self.device))
        dot_product_total = defaultdict(lambda: torch.zeros((), device=self.device))

        for name, p in self.named_parameters():
            if p is None or p.data is None:
                continue

            key = name.split(".")[1]
            param_sq[key] = param_sq[key] + (p.detach().float() ** 2).sum()

            if p.grad is not None:
                g = p.grad.detach().float() if p.grad.dtype != torch.float32 else p.grad.detach()
                grad_sq[key] = grad_sq[key] + (g ** 2).sum()

                if p in optimizer.state and "exp_avg" in optimizer.state[p]:
                    m = optimizer.state[p]["exp_avg"].detach()
                    m = m.float() if m.dtype != torch.float32 else m

                    dot_product_total[key] = dot_product_total[key] + (g * m).sum()
                    momentum_sq[key] = momentum_sq[key] + (m ** 2).sum()

        total_grad = torch.stack(
            list(grad_sq.values()) or [torch.zeros((), device=self.device)]
        ).sum().sqrt()

        metrics = {"grad/total": total_grad}

        eps = 1e-12
        total_dot = torch.zeros((), device=self.device)
        total_grad_sq = torch.zeros((), device=self.device)
        total_momentum_sq = torch.zeros((), device=self.device)

        for k in sorted(grad_sq.keys()):
            gnorm = grad_sq[k].sqrt()
            pnorm = param_sq[k].sqrt().clamp_min(eps)

            metrics[f"grad/{k}"] = gnorm
            metrics[f"gradratio/{k}"] = gnorm / pnorm
            metrics[f"pnorm/{k}"] = pnorm

            if momentum_sq[k] > 0:
                per_layer_alignment = dot_product_total[k] / (
                    gnorm * momentum_sq[k].sqrt() + eps
                )
                metrics[f"grad_alignment/{k}"] = per_layer_alignment

            total_dot = total_dot + dot_product_total[k]
            total_grad_sq = total_grad_sq + grad_sq[k]
            total_momentum_sq = total_momentum_sq + momentum_sq[k]

        if total_momentum_sq > 0:
            total_alignment = total_dot / (
                total_grad_sq.sqrt() * total_momentum_sq.sqrt() + eps
            )
            metrics["grad_alignment/total"] = total_alignment

        self.log_dict(
            metrics,
            on_step=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

        return super().on_before_optimizer_step(optimizer)

    def on_train_batch_start(self, batch, batch_idx):
        self.tic = datetime.datetime.now()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        toc = datetime.datetime.now()
        dt = (toc - self.tic).total_seconds()

        self.log("dt", dt, on_step=True)

        self.min_dt = min(dt, self.min_dt)
        self.log("min_dt", self.min_dt, on_step=True)