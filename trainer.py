import logging
import re
import time
from collections import defaultdict, OrderedDict

import numpy as np
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
import omegaconf.dictconfig
import torch
import torch.nn as nn

from data.datamodule import Era5DataModule
from model.paradis import Paradis
from utils.loss import ParadisLoss
from utils.normalization import denormalize_humidity, denormalize_precipitation
from utils.postprocessing import denormalize_datasets, convert_cartesian_to_spherical_winds
from utils.file_output import ZarrForecastWriter
from utils.crps_loss import (
    TwoMemberAlmostFairCRPS,
    TwoMemberSpectralAlmostFairCRPS,
)

# Configure torch.compile to handle dynamic shapes in Muon/NorMuon optimizers
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.force_parameter_static_shapes = False


def build_param_groups(model, lr, weight_decay, optimizer_name):
    muon_params = []
    adamw_params = []
    seen = set()

    for module_name, module in model.named_modules():
        # Muon: standard hidden-layer weights
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if (
                getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            ):
                muon_params.append(module.weight)
                seen.add(id(module.weight))

            if getattr(module, "bias", None) is not None and module.bias.requires_grad:
                adamw_params.append(module.bias)
                seen.add(id(module.bias))

    # Everything else defaults to AdamW
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        adamw_params.append(p)
        seen.add(id(p))

    return [
        dict(
            params=muon_params,
            algorithm=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            flatten=True,
        ),
        dict(
            params=adamw_params,
            algorithm="adamw",
            lr=lr,
            weight_decay=weight_decay,
        ),
    ]


def _strip_orig_mod_prefix(state_dict: dict[str, torch.Tensor]) -> OrderedDict:
    fixed = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("._orig_mod.", ".")
        # also handle rare case where the key starts with "_orig_mod."
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod.") :]
        fixed[new_k] = v
    return fixed

def _extract_model_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict:
    model_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_k = k.replace("._orig_mod.", ".")

        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]

        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
            model_state_dict[new_k] = v

    return model_state_dict


class LitParadis(L.LightningModule):
    """Lightning module for Paradis model training."""

    model: torch.nn.Module

    def __init__(
        self, datamodule: Era5DataModule, cfg: omegaconf.dictconfig.DictConfig
    ) -> None:
        """Initialize the training module.

        Args:
            datamodule: Lightning datamodule containing dataset information
            cfg: Model configuration dictionary
        """
        super().__init__()

        # Instantiate the model
        self.min_dt = 1e10
        self.datamodule = datamodule
        lat_grid = datamodule.dataset.lat_rad_grid
        lon_grid = datamodule.dataset.lon_rad_grid
        self.model = Paradis(datamodule, cfg, lat_grid, lon_grid)
        self.cfg = cfg
        self.n_inputs = cfg.dataset.n_time_inputs
        
        # ------------------------------------------------------------------ #
        # Probabilistic ensemble configuration
        # ------------------------------------------------------------------ #

        self.noise_channels = cfg.model.noise_channels
        self.num_members = cfg.training.get("num_ensemble_members", 2)

        self.nlat = lat_grid.shape[0]
        self.nlon = lat_grid.shape[1]

        if self.noise_channels <= 0:
            raise ValueError(
                f"Probabilistic Paradis requires noise_channels > 0, "
                f"got noise_channels={self.noise_channels}."
            )

        if self.num_members != 2 and not cfg.forecast.enable:
            raise ValueError(
                "Two-member afCRPS training requires "
                f"num_ensemble_members=2, got {self.num_members}."
            )

        # Log metrics
        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        if self.global_rank == 0:
            logging.info("Number of trainable parameters: {:,}".format(num_parameters))
            logging.info(
                f"Probabilistic ensemble training: "
                f"{self.num_members} members, "
                f"noise_channels={self.noise_channels}"
            )
            
        # Access output_name_order from configuration
        self.output_name_order = datamodule.output_name_order

        num_levels = len(cfg.features.pressure_levels)

        # Construct variable loss weight tensor from YAML configuration
        atmospheric_weights = torch.tensor(
            [
                cfg.training.variable_loss_weights.atmospheric[var]
                for var in cfg.features.output.atmospheric
            ],
            dtype=torch.float32,
        )

        surface_weights = torch.tensor(
            [
                cfg.training.variable_loss_weights.surface[var]
                for var in cfg.features.output.surface
            ],
            dtype=torch.float32,
        )

        # Create a mapping of variable names to their weights
        atmospheric_vars = cfg.features.output.atmospheric
        surface_vars = cfg.features.output.surface
        var_name_to_weight = {
            **{var: atmospheric_weights[i] for i, var in enumerate(atmospheric_vars)},
            **{var: surface_weights[i] for i, var in enumerate(surface_vars)},
        }

        # Initialize reordered weights tensor
        num_features = len(atmospheric_weights) * num_levels + len(surface_weights)

        var_loss_weights_reordered = torch.zeros(
            datamodule.num_out_features,
            dtype=torch.float32,
        )

        for i, feature in enumerate(datamodule.output_name_order):
            var_name = re.sub(r"_h\d+$", "", feature)

            if var_name in cfg.training.variable_loss_weights.atmospheric:
                var_loss_weights_reordered[i] = (
                    cfg.training.variable_loss_weights.atmospheric[var_name]
                )
            elif var_name in cfg.training.variable_loss_weights.surface:
                var_loss_weights_reordered[i] = (
                    cfg.training.variable_loss_weights.surface[var_name]
                )
            else:
                raise ValueError(
                    f"No loss weight configured for output feature '{feature}' "
                    f"(base variable '{var_name}')."
                )

        if var_loss_weights_reordered.numel() != datamodule.num_out_features:
            raise ValueError(
                f"Loss weight count mismatch: got {var_loss_weights_reordered.numel()}, "
                f"expected {datamodule.num_out_features}."
            )

        # Initialize loss function with delta schedule parameters
        if not cfg.forecast.enable:
            # ------------------------------------------------------------------ #
            # Probabilistic loss
            # ------------------------------------------------------------------ #

            alpha = cfg.training.get("crps_alpha", 0.95)

            pairwise_coeff = cfg.training.get("crps_pairwise_coeff", None,)

            self.crps_loss = TwoMemberAlmostFairCRPS(
                var_loss_weights=var_loss_weights_reordered, lat_grid=datamodule.lat, alpha=alpha,
                pairwise_coeff=pairwise_coeff, apply_latitude_weights=cfg.training.loss_function.lat_weights,
            )
            
            spectral_cfg = cfg.training.get("spectral_crps", {},)
            self.spectral_crps_weight = float(spectral_cfg.get("weight", 0.0))

            if self.spectral_crps_weight > 0.0:
                self.spectral_crps_loss = (
                    TwoMemberSpectralAlmostFairCRPS(
                        nlat=self.nlat,
                        nlon=self.nlon,
                        var_loss_weights=var_loss_weights_reordered,
                        alpha=alpha,
                        pairwise_coeff=pairwise_coeff,
                        grid=spectral_cfg.get(
                            "grid",
                            "equiangular",
                        ),
                    )
                )
            else:
                self.spectral_crps_loss = None
                
            self.loss_fn = ParadisLoss(
                loss_function="mse",
                lat_grid=datamodule.lat,
                pressure_levels=torch.tensor(
                    cfg.features.pressure_levels,
                    dtype=torch.float32,
                ),
                num_features=datamodule.num_out_features,
                num_surface_vars=len(
                    cfg.features.output.surface
                ),
                var_loss_weights=var_loss_weights_reordered,
                output_name_order=datamodule.output_name_order,
                delta_loss=cfg.training.loss_function.delta_loss,
                apply_latitude_weights=cfg.training.loss_function.lat_weights,
            )

            self.automatic_optimization = False

        self.num_common_features = datamodule.num_common_features
        self.print_losses = cfg.training.print_losses

        # Load weights only but reset lightning configuration
        if cfg.init.checkpoint_path and not cfg.init.restart and not cfg.forecast.enable:
            # Load into CPU, then Lightning will transfer to GPU
            checkpoint = torch.load(
                cfg.init.checkpoint_path, weights_only=False, map_location="cpu"
            )

            sd = checkpoint["state_dict"]

            # # Make sure model can read checkpoint whether it has been previously compiled or not
            sd = _strip_orig_mod_prefix(sd)

            # # # Interpolate GlobalBias parameters for resolution change
            # # # We look for any keys ending in .U or .V belonging to a GlobalBias module
            for k in list(sd.keys()):
                if k.endswith(".U") or k.endswith(".V"):
                    old_param = sd[k]  # Shape: [rank, old_size]

                    # Determine target size from the current model's parameters
                    # self.state_dict() reflects the NEW 0.25 degree initialization
                    target_size = self.state_dict()[k].shape[-1]

                    if old_param.shape[-1] != target_size:
                        # unsqueeze to [1, rank, old_size] for F.interpolate
                        # then interpolate to [1, rank, target_size]
                        new_param = torch.nn.functional.interpolate(
                            old_param.unsqueeze(0),
                            size=target_size,
                            mode="linear",
                            align_corners=True,
                        ).squeeze(0)

                        sd[k] = new_param
                        print(
                            f"Interpolated {k}: {old_param.shape[-1]} -> {target_size}"
                        )

            self.load_state_dict(sd, strict=True)
            
        if cfg.forecast.enable and cfg.init.checkpoint_path:
            checkpoint = torch.load(cfg.init.checkpoint_path, weights_only=False, map_location="cpu")
            sd = _extract_model_state_dict(checkpoint["state_dict"])
            self.model.load_state_dict(sd, strict=True)
            logging.info(f"Loaded forecast model weights from {cfg.init.checkpoint_path}")

        # Compile model in place
        if cfg.compute.compile == True:
            self.model.compile(
                mode="default",
                fullgraph=True,
                dynamic=False,
                backend="inductor",
            )
        elif cfg.compute.compile == "modules":
            self.model._compile()

        self.epoch_start_time = None

        # Store the index and stats of the report quantities
        self.enable_reports = cfg.training.reports.enable
        if not cfg.forecast.enable and self.enable_reports:
            self.report_features = cfg.training.reports.features
            self.report_ind = [
                datamodule.dataset.dyn_input_features.index(feature)
                for feature in cfg.training.reports.features
            ]
            self.report_ind = torch.tensor(self.report_ind, dtype=torch.long)
            self.report_mean = torch.from_numpy(datamodule.dataset.report_stats["mean"])
            self.report_std = torch.from_numpy(datamodule.dataset.report_stats["std"])

        self.custom_norms = not cfg.normalization.standard
        self.log_statistics = cfg.training.get("log_additional_stats", False)

        if cfg.forecast.enable:
            self.forecast_writer = ZarrForecastWriter(cfg, datamodule.dataset)

    def _get_report_rmse(self, output_data, pred_data):
        lat_weights = self.loss_fn.lat_weights.view(1, 1, -1, 1).to(output_data.device)

        errors = torch.empty(
            len(self.report_ind), dtype=output_data.dtype, device=output_data.device
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
    
    def _sample_raw_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype, ) -> torch.Tensor:

        return torch.randn(batch_size, self.noise_channels, self.nlat, self.nlon, device=device, dtype=dtype,)

    def _embed_raw_noise(self, raw_noise: torch.Tensor,) -> torch.Tensor:

        B, C_n, lat, lon = raw_noise.shape
        noise_flat = (raw_noise.permute(0, 2, 3, 1).reshape(B * lat * lon, C_n))
        emb_flat = self.model.noise_embedding(noise_flat)
        emb_dim = emb_flat.shape[-1]
        noise_emb = (emb_flat.reshape(B, lat, lon, emb_dim).permute(0, 3, 1, 2))

        return noise_emb
    
    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        return self.model(x, noise_emb=noise_emb)

    def configure_optimizers(self):  # type: ignore
        """Configure optimizer and learning rate scheduler."""
        cfg = self.cfg.training

        if cfg.optimizer.name == "adamw":

            param_groups = self.model.parameters()
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            )

        elif "muon" in cfg.optimizer.name:
            from dion import Muon, NorMuon

            param_groups = build_param_groups(
                self.model,
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                optimizer_name=cfg.optimizer.name,
            )

            if cfg.optimizer.name == "muon":
                optimizer = Muon(
                    param_groups,
                    lr=cfg.optimizer.lr,
                    weight_decay=cfg.optimizer.weight_decay,
                    betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                    use_triton=True,
                )
            elif cfg.optimizer.name == "normuon":
                optimizer = NorMuon(
                    param_groups,
                    lr=cfg.optimizer.lr,
                    weight_decay=cfg.optimizer.weight_decay,
                    betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                    use_triton=True,
                )
            else:
                raise ValueError(f"Optimizer {cfg.optimizer.name} not supported. Choose between normuon|muon")

        enabled_schedulers = sum(
            [
                cfg.scheduler.one_cycle.enabled,
                cfg.scheduler.reduce_lr.enabled,
                cfg.scheduler.wsd.enabled,
            ]
        )

        # Ensure only one is enabled
        if enabled_schedulers != 1:
            raise ValueError(
                f"Invalid config: Exactly one scheduler must "
                + f"be enabled, but found {enabled_schedulers} enabled."
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
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        elif cfg.scheduler.reduce_lr.enabled:

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
                    "monitor": "val_loss_epoch",  # Monitor epoch-level validation loss
                    "interval": "epoch",  # When the scheduler should make decisions
                    "frequency": 1,
                },
            }
        elif cfg.scheduler.wsd.enabled:
            total_steps = self.trainer.estimated_stepping_batches

            # Set warmup and decay periods
            if cfg.scheduler.wsd.warmup >= 1:
                # Value >= 1, so it's a number of steps
                warmup_steps = cfg.scheduler.wsd.warmup
            else:
                warmup_steps = cfg.scheduler.wsd.warmup * total_steps

            if cfg.scheduler.wsd.decay >= 1:
                # Value >= 1, so it's a number of steps
                decay_steps = cfg.scheduler.wsd.decay
            else:
                decay_steps = cfg.scheduler.wsd.decay * total_steps

            # Sanity checks
            assert warmup_steps >= 0
            assert decay_steps >= 0
            assert warmup_steps + decay_steps <= total_steps

            steady_steps = total_steps - (warmup_steps + decay_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    # Increasing learning rate phase
                    return (step + 1) / warmup_steps
                elif step <= warmup_steps + steady_steps:
                    # Constant learning rate
                    return 1.0
                else:
                    # Decay learning rate
                    decay_ratio = (total_steps - step) / decay_steps
                    return decay_ratio  # Linear decay

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            # No known scheduler was active
            active_schedulers = [
                k for (k, v) in cfg.scheduler.items() if "enabled" in v and v["enabled"]
            ]
            if len(active_schedulers) == 0:
                # Should not happen if enabled_schedulers check above is still present
                raise ValueError(f"No scheduler activated")
            else:
                raise ValueError(
                    f'Unknown schedule activated: {", ".join(active_schedulers)}'
                )

    @rank_zero_only
    def on_fit_start(self):
        total = sum(p.numel() for p in self.parameters())

        if self.logger and hasattr(self.logger, "experiment"):
            tb = self.logger.experiment
            tb.add_scalar("model/num_parameters", total, global_step=0)

    def on_predict_start(self):
        if not self.cfg.forecast.enable:
            return

        if self.trainer.is_global_zero and not self.forecast_writer.store_initialized:
            self.forecast_writer.init_store(self.datamodule.dataset)

        self.trainer.strategy.barrier()

    def on_train_epoch_start(self):
        """Record the start time of the epoch."""
        if self.print_losses:
            self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):

        input_data, true_data, forcings, constant_data = batch
        opt = self.optimizers()
        grad_accum_steps = self.cfg.training.get("accumulate_grad_batches", 1)

        if batch_idx % grad_accum_steps == 0:
            opt.zero_grad(set_to_none=True)

        constants = constant_data[:, :1].permute(0, 1, 4, 2, 3)
        forcings = forcings.permute(0, 1, 4, 2, 3)

        batch_size = input_data.size(0)
        num_steps = true_data.size(1)

        member1_input = input_data.clone()
        member2_input = input_data.clone()

        log_spatial_loss = input_data.new_zeros(())
        log_fit = input_data.new_zeros(())
        log_spread = input_data.new_zeros(())
        log_spectral_loss = input_data.new_zeros(())
        log_total_loss = input_data.new_zeros(())

        for step in range(num_steps):

            y = true_data[:, step]

            forcings_step = forcings[:, step].unsqueeze(1)

            model_input1 = torch.cat([member1_input, forcings_step, constants], dim=2).squeeze(1)
            model_input2 = torch.cat([member2_input, forcings_step, constants], dim=2).squeeze(1)

            raw_noise1 = self._sample_raw_noise(batch_size=batch_size, device=input_data.device, dtype=input_data.dtype)
            raw_noise2 = self._sample_raw_noise(batch_size=batch_size, device=input_data.device, dtype=input_data.dtype)

            # ---------------------------------------------------------- #
            # Member 1: fit gradient
            # ---------------------------------------------------------- #

            noise1 = self._embed_raw_noise(raw_noise1)
            x1 = self.forward(model_input1, noise_emb=noise1)
            loss_x1_fit = self.crps_loss.fit_term(x1, y)

            if self.spectral_crps_loss is not None:
                loss_x1_fit = loss_x1_fit + self.spectral_crps_weight * self.spectral_crps_loss.fit_term(x1, y)

            loss_x1_fit = loss_x1_fit / num_steps / grad_accum_steps
            self.manual_backward(loss_x1_fit)
            x1_det = x1.detach()

            del x1
            del noise1
            del loss_x1_fit

            # ---------------------------------------------------------- #
            # Member 2: fit + spread gradient
            # ---------------------------------------------------------- #

            noise2 = self._embed_raw_noise(raw_noise2)
            x2 = self.forward(model_input2, noise_emb=noise2)
            fit_x2 = self.crps_loss.fit_term(x2, y)
            spread_x2 = self.crps_loss.spread_term(x1_det, x2)
            loss_x2 = fit_x2 - spread_x2

            if self.spectral_crps_loss is not None:
                spec_fit_x2 = self.spectral_crps_loss.fit_term(x2, y)
                spec_spread_x2 = self.spectral_crps_loss.spread_term(x1_det, x2)
                loss_x2 = loss_x2 + self.spectral_crps_weight * (spec_fit_x2 - spec_spread_x2)

            loss_x2 = loss_x2 / num_steps / grad_accum_steps
            self.manual_backward(loss_x2)
            x2_det = x2.detach()

            del x2
            del noise2
            del fit_x2
            del spread_x2
            del loss_x2

            # ---------------------------------------------------------- #
            # Recompute member 1: spread gradient
            # ---------------------------------------------------------- #

            noise1_re = self._embed_raw_noise(raw_noise1)
            x1_re = self.forward(model_input1, noise_emb=noise1_re)
            spread_x1 = self.crps_loss.spread_term(x1_re, x2_det)
            loss_x1_spread = -spread_x1

            if self.spectral_crps_loss is not None:
                spec_spread_x1 = self.spectral_crps_loss.spread_term(x1_re, x2_det)
                loss_x1_spread = loss_x1_spread - self.spectral_crps_weight * spec_spread_x1

            loss_x1_spread = loss_x1_spread / num_steps / grad_accum_steps
            self.manual_backward(loss_x1_spread)
            x1_re_det = x1_re.detach()

            del x1_re
            del noise1_re
            del spread_x1
            del loss_x1_spread

            # ---------------------------------------------------------- #
            # Logging
            # ---------------------------------------------------------- #

            with torch.no_grad():
                step_spatial_loss, step_fit, step_spread = self.crps_loss.full_loss_for_logging(x1_re_det, x2_det, y)
                weighted_spectral_loss = input_data.new_zeros(())
                step_total_loss = step_spatial_loss

                if self.spectral_crps_loss is not None:
                    spectral_loss, _, _ = self.spectral_crps_loss.full_loss_for_logging(x1_re_det, x2_det, y)
                    weighted_spectral_loss = self.spectral_crps_weight * spectral_loss
                    step_total_loss = step_total_loss + weighted_spectral_loss

                log_spatial_loss += step_spatial_loss
                log_fit += step_fit
                log_spread += step_spread
                log_spectral_loss += weighted_spectral_loss
                log_total_loss += step_total_loss

            # ---------------------------------------------------------- #
            # Independent autoregressive propagation
            # ---------------------------------------------------------- #

            member1_input = self._autoregression_next_input(model_input1, x1_re_det).unsqueeze(1)
            member2_input = self._autoregression_next_input(model_input2, x2_det).unsqueeze(1)

            member1_input = member1_input.detach()
            member2_input = member2_input.detach()

            del x1_det
            del x1_re_det
            del x2_det
            del raw_noise1
            del raw_noise2

        should_step_optimizer = (batch_idx + 1) % grad_accum_steps == 0 or self.trainer.is_last_batch

        if should_step_optimizer:
            opt.step()
            scheduler = self.lr_schedulers()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    pass
                else:
                    scheduler.step()

        batch_spatial_loss = log_spatial_loss / num_steps
        batch_fit = log_fit / num_steps
        batch_spread = log_spread / num_steps
        batch_spectral_loss = log_spectral_loss / num_steps
        batch_total_loss = log_total_loss / num_steps

        self.log("train_spatial_afcrps_loss", batch_spatial_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_spatial_afcrps_fit", batch_fit, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_spatial_afcrps_spread", batch_spread, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_spectral_afcrps_loss", batch_spectral_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_total_afcrps_loss", batch_total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("lr", opt.param_groups[0]["lr"], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("forecast_steps", num_steps, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return batch_total_loss.detach()
    

    def validation_step(self, batch, batch_idx):

        input_data, true_data, forcings, constant_data = batch

        constants = constant_data[:, :1].permute(0, 1, 4, 2, 3)
        forcings = forcings.permute(0, 1, 4, 2, 3)

        batch_size = input_data.size(0)
        num_steps = true_data.size(1)

        member1_input = input_data.clone()
        member2_input = input_data.clone()

        total_spatial_loss = input_data.new_zeros(())
        total_fit = input_data.new_zeros(())
        total_spread = input_data.new_zeros(())
        total_spectral_loss = input_data.new_zeros(())
        total_loss = input_data.new_zeros(())

        if self.enable_reports:
            report_loss = torch.zeros(len(self.report_ind), dtype=input_data.dtype, device=input_data.device)
            
        for step in range(num_steps):

            y = true_data[:, step]

            forcings_step = forcings[:, step].unsqueeze(1)

            model_input1 = torch.cat([member1_input, forcings_step, constants], dim=2).squeeze(1)
            model_input2 = torch.cat([member2_input, forcings_step, constants], dim=2).squeeze(1)

            raw_noise1 = self._sample_raw_noise(batch_size=batch_size, device=input_data.device, dtype=input_data.dtype)
            raw_noise2 = self._sample_raw_noise(batch_size=batch_size, device=input_data.device, dtype=input_data.dtype)

            noise1 = self._embed_raw_noise(raw_noise1)
            noise2 = self._embed_raw_noise(raw_noise2)

            x1 = self.forward(model_input1, noise_emb=noise1)
            x2 = self.forward(model_input2, noise_emb=noise2)

            # ---------------------------------------------------------- #
            # Spatial afCRPS
            # ---------------------------------------------------------- #

            step_spatial_loss, step_fit, step_spread = self.crps_loss.full_loss_for_logging(x1, x2, y)

            step_total_loss = step_spatial_loss

            # ---------------------------------------------------------- #
            # Spectral afCRPS
            # ---------------------------------------------------------- #

            weighted_spectral_loss = input_data.new_zeros(())

            if self.spectral_crps_loss is not None:
                spectral_loss, _, _ = self.spectral_crps_loss.full_loss_for_logging(x1, x2, y)
                weighted_spectral_loss = self.spectral_crps_weight * spectral_loss
                step_total_loss = step_total_loss + weighted_spectral_loss

            total_spatial_loss += step_spatial_loss
            total_fit += step_fit
            total_spread += step_spread
            total_spectral_loss += weighted_spectral_loss
            total_loss += step_total_loss

            # ---------------------------------------------------------- #
            # Ensemble-mean RMSE reporting
            # ---------------------------------------------------------- #

            ensemble_mean = 0.5 * (x1 + x2)
                
            if self.enable_reports:
                report_loss += self._get_report_rmse(ensemble_mean, y)

            # ---------------------------------------------------------- #
            # Independent autoregressive propagation
            # ---------------------------------------------------------- #

            member1_input = self._autoregression_next_input(model_input1, x1).unsqueeze(1)
            member2_input = self._autoregression_next_input(model_input2, x2).unsqueeze(1)

            member1_input = member1_input.detach()
            member2_input = member2_input.detach()

        batch_spatial_loss = total_spatial_loss / num_steps
        batch_fit = total_fit / num_steps
        batch_spread = total_spread / num_steps
        batch_spectral_loss = total_spectral_loss / num_steps
        batch_total_loss = total_loss / num_steps

        # -------------------------------------------------------------- #
        # Primary validation loss = CRPS objective
        # -------------------------------------------------------------- #

        self.log("val_loss", batch_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log("val_spatial_afcrps_loss", batch_spatial_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_spatial_afcrps_fit", batch_fit, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_spatial_afcrps_spread", batch_spread, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_spectral_afcrps_loss", batch_spectral_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_total_afcrps_loss", batch_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # -------------------------------------------------------------- #
        # RMSE reports using ensemble mean
        # -------------------------------------------------------------- #
        if self.enable_reports:
            for i, name in enumerate(self.cfg.training.reports.features):
                self.log(name, report_loss[i] / num_steps, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return batch_total_loss

    def _autoregression_next_input(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor,
    ) -> torch.Tensor:
        """Update autoregressive channel stack."""
        common = output_data[:, : self.num_common_features]

        if self.n_inputs == 1:
            return common

        lag_channels = self.num_common_features * self.n_inputs

        return torch.cat(
            [
                input_data[:, self.num_common_features : lag_channels],
                common,
            ],
            dim=1,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        sample_indices, input_data, forcings, constant_data = batch
        dataset = self.datamodule.dataset

        num_forecast_steps = self.cfg.model.forecast_steps
        output_frequency = self.cfg.forecast.output_frequency
        write_every_n = self.cfg.forecast.get("write_every_n", num_forecast_steps)
        num_members = int(self.cfg.forecast.num_ensemble_members)

        if num_members <= 0:
            raise ValueError(f"forecast.num_ensemble_members must be > 0, got {num_members}")

        output_features = list(dataset.dyn_output_features)

        constants = constant_data[:, :1].permute(0, 1, 4, 2, 3)

        batch_size = input_data.size(0)

        # Each ensemble member must maintain its own autoregressive state.
        member_inputs = [input_data.clone() for _ in range(num_members)]

        chunk_buffer = []
        chunk_start_idx = None
        stored_step_idx = 0

        for step in range(num_forecast_steps):

            forcings_step = forcings[:, step].unsqueeze(1).permute(0, 1, 4, 2, 3)

            step_member_outputs = []

            for member in range(num_members):

                model_input = torch.cat([member_inputs[member], forcings_step, constants], dim=2).squeeze(1)

                raw_noise = self._sample_raw_noise(batch_size=batch_size, device=model_input.device, dtype=model_input.dtype)
                noise_emb = self._embed_raw_noise(raw_noise)

                output_data = self(model_input, noise_emb=noise_emb)

                member_inputs[member] = self._autoregression_next_input(model_input, output_data).unsqueeze(1).detach()

                step_member_outputs.append(output_data.detach())

                del raw_noise
                del noise_emb
                del output_data

            # Shape:
            #   [B, M, F, Lat, Lon]
            step_output = torch.stack(step_member_outputs, dim=1)

            if step % output_frequency == 0:

                if chunk_start_idx is None:
                    chunk_start_idx = stored_step_idx

                chunk_buffer.append(step_output)
                stored_step_idx += 1

                if len(chunk_buffer) == write_every_n:

                    # Shape:
                    #   [B, M, T_chunk, F, Lat, Lon]
                    chunk_tensor = torch.stack(chunk_buffer, dim=2).cpu()

                    # denormalize_datasets expects:
                    #   [B, T_chunk, F, Lat, Lon]
                    # so process each ensemble member separately.
                    for member in range(num_members):
                        denormalize_datasets(None, chunk_tensor[:, member], dataset)

                    chunk_np = chunk_tensor.numpy()

                    # Wind conversion also expects one forecast trajectory at a time.
                    for member in range(num_members):
                        member_forecast = np.ascontiguousarray(chunk_np[:, member])

                        convert_cartesian_to_spherical_winds(
                            dataset.lat,
                            dataset.lon,
                            self.cfg,
                            member_forecast,
                            output_features,
                        )

                        chunk_np[:, member] = member_forecast

                    self.forecast_writer.write_forecast_chunk(
                        forecast=chunk_np,
                        sample_indices=None if sample_indices is None else sample_indices.cpu().numpy(),
                        start_idx=chunk_start_idx,
                        dataset=dataset,
                    )

                    del chunk_tensor
                    del chunk_np

                    chunk_buffer.clear()
                    chunk_start_idx = None

            del step_member_outputs
            del step_output

        # Write any remaining forecast outputs.
        if chunk_buffer:

            # Shape:
            #   [B, M, T_chunk, F, Lat, Lon]
            chunk_tensor = torch.stack(chunk_buffer, dim=2).cpu()

            for member in range(num_members):
                denormalize_datasets(None, chunk_tensor[:, member], dataset)

            chunk_np = chunk_tensor.numpy()

            for member in range(num_members):
                member_forecast = np.ascontiguousarray(chunk_np[:, member])

                convert_cartesian_to_spherical_winds(
                    dataset.lat,
                    dataset.lon,
                    self.cfg,
                    member_forecast,
                    output_features,
                )

                chunk_np[:, member] = member_forecast

            self.forecast_writer.write_forecast_chunk(
                forecast=chunk_np,
                sample_indices=None if sample_indices is None else sample_indices.cpu().numpy(),
                start_idx=chunk_start_idx,
                dataset=dataset,
            )

            del chunk_tensor
            del chunk_np

        return None

    def on_train_epoch_end(self):
        """Log epoch time and metrics if printing losses."""
        if self.print_losses and self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

            # Get the losses using the logged metrics
            train_loss = self.trainer.callback_metrics.get("train_total_afcrps_loss")
            val_loss = self.trainer.callback_metrics.get("val_loss")

            if (
                self.trainer.is_global_zero
                and train_loss is not None
                and val_loss is not None
            ):
                print(
                    f"Epoch {self.current_epoch:4d} | "
                    f"Train afCRPS: {train_loss.item():.6f} | "
                    f"Val afCRPS: {val_loss.item():.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Elapsed time: {elapsed_time:.4f}s"
                )

    def on_train_end(self):
        """Called when training ends."""
        logging.info(f"Training completed after {self.current_epoch + 1} epochs")

    def on_before_optimizer_step(self, optimizer):

        if self.log_statistics:
            grad_sq = defaultdict(lambda: torch.zeros((), device=self.device))
            param_sq = defaultdict(lambda: torch.zeros((), device=self.device))
            momentum_sq = defaultdict(lambda: torch.zeros((), device=self.device))
            dot_product_total = defaultdict(lambda: torch.zeros((), device=self.device))

            for name, p in self.named_parameters():
                if p is None or p.data is None:
                    continue
                key = name.split(".")[1]

                # param norm (for grad/param ratio)
                param_sq[key] = param_sq[key] + (p.detach().float() ** 2).sum()

                # grad norm
                if p.grad is not None:
                    g = p.grad.detach()
                    if g.dtype != torch.float32:
                        g = g.float()
                    grad_sq[key] = grad_sq[key] + (g**2).sum()

                    # Compute grad-momentum alignment (cosine similarity)
                    if p in optimizer.state and "exp_avg" in optimizer.state[p]:
                        m = optimizer.state[p]["exp_avg"].detach()
                        if m.dtype != torch.float32:
                            m = m.float()

                        # Accumulate for cosine similarity computation
                        dot_product_total[key] = dot_product_total[key] + (g * m).sum()
                        momentum_sq[key] = momentum_sq[key] + (m**2).sum()

            total_grad = (
                torch.stack(
                    list(grad_sq.values()) or [torch.zeros((), device=self.device)]
                )
                .sum()
                .sqrt()
            )

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

                # Add grad-momentum alignment metrics (per-layer cosine similarity)
                if momentum_sq[k] > 0:
                    g_norm = grad_sq[k].sqrt()
                    m_norm = momentum_sq[k].sqrt()
                    per_layer_alignment = dot_product_total[k] / (g_norm * m_norm + eps)
                    metrics[f"grad_alignment/{k}"] = per_layer_alignment

                # Accumulate for total cosine similarity
                total_dot = total_dot + dot_product_total[k]
                total_grad_sq = total_grad_sq + grad_sq[k]
                total_momentum_sq = total_momentum_sq + momentum_sq[k]

            # Compute overall grad-momentum alignment (true cosine similarity)
            if total_momentum_sq > 0:
                total_grad_norm = total_grad_sq.sqrt()
                total_momentum_norm = total_momentum_sq.sqrt()
                total_alignment = total_dot / (
                    total_grad_norm * total_momentum_norm + eps
                )
                metrics["grad_alignment/total"] = total_alignment

            self.log_dict(
                metrics, on_step=True, logger=True, prog_bar=False, sync_dist=True
            )

        return super().on_before_optimizer_step(optimizer)

    def on_train_batch_start(self, batch, batch_idx):
        # Record current time for time-per-step calculation
        self.tic = time.perf_counter()

        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        dt = time.perf_counter() - self.tic

        self.log(
            "dt",
            dt,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        return super().on_train_batch_end(outputs, batch, batch_idx)
