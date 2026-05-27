"""Checkpoint loading utilities for PARADIS / PARADIS-ENS."""

import logging
from collections import Counter

import torch


def read_checkpoint_state_dict(checkpoint_path: str) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cpu",
    )
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def load_checkpoint_for_litmodel(litmodel, cfg) -> None:
    """Load checkpoint into LitParadis.

    Supports:
        cfg.init.checkpoint_type = "ensemble"
        cfg.init.checkpoint_type = "deterministic"
    """

    checkpoint_type = cfg.init.get("checkpoint_type", "ensemble")

    if checkpoint_type == "ensemble":
        load_ensemble_checkpoint(
            litmodel=litmodel,
            checkpoint_path=cfg.init.checkpoint_path,
        )

    elif checkpoint_type == "deterministic":
        load_deterministic_checkpoint_into_ensemble(
            litmodel=litmodel,
            checkpoint_path=cfg.init.checkpoint_path,
        )

    else:
        raise ValueError(
            f"Unknown cfg.init.checkpoint_type={checkpoint_type}. "
            "Use 'ensemble' or 'deterministic'."
        )


def load_ensemble_checkpoint(litmodel, checkpoint_path: str) -> None:
    """Normal strict loading for an ensemble checkpoint."""

    state_dict = read_checkpoint_state_dict(checkpoint_path)
    litmodel.load_state_dict(state_dict, strict=True)

    if litmodel.global_rank == 0:
        logging.info(f"Loaded ensemble checkpoint from: {checkpoint_path}")


def load_deterministic_checkpoint_into_ensemble(
    litmodel,
    checkpoint_path: str,
) -> None:
    """Initialize ensemble model from deterministic checkpoint.

    This function:
    1. directly copies parameters whose names and shapes are unchanged;
    2. maps deterministic ChannelNorm parameters to ensemble cond_norm parameters;
    3. maps renamed block-body parameters;
    4. neutralizes noise-conditioning layers so the ensemble initially behaves
       as close as possible to the deterministic model.
    """

    det_state = read_checkpoint_state_dict(checkpoint_path)

    ens_state = litmodel.state_dict()
    new_state = dict(ens_state)

    directly_copied = []
    norm_mapped = []
    body_mapped = []
    noise_neutralized = []

    # -------------------------------------------------------------- #
    # 1. Direct copy for keys that still match exactly
    # -------------------------------------------------------------- #
    for k, v in det_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
            directly_copied.append(k)

    # -------------------------------------------------------------- #
    # 2. Explicit ChannelNorm -> cond_norm mapping
    # -------------------------------------------------------------- #
    for k, v in det_state.items():
        new_k = None

        if k.endswith(".0-ChannelNorm.weight"):
            new_k = k.replace(".0-ChannelNorm.weight", ".cond_norm.weight")

        elif k.endswith(".0-ChannelNorm.bias"):
            new_k = k.replace(".0-ChannelNorm.bias", ".cond_norm.bias")

        if new_k is not None:
            if new_k in new_state and new_state[new_k].shape == v.shape:
                new_state[new_k] = v
                norm_mapped.append((k, new_k))

    # -------------------------------------------------------------- #
    # 3. Map renamed block-body parameters
    # -------------------------------------------------------------- #
    for k, v in det_state.items():
        candidate_keys = []

        # velocity_nets / diffusion / reaction:
        # old: model.velocity_nets.0.0-SepConv...
        # new: model.velocity_nets.0.body.0-SepConv...
        for block_name in ["velocity_nets", "diffusion", "reaction"]:
            prefix = f"model.{block_name}."

            if k.startswith(prefix):
                parts = k.split(".")

                if len(parts) >= 4 and parts[3].startswith(("0-", "1-")):
                    new_parts = parts[:3] + ["body"] + parts[3:]
                    candidate_keys.append(".".join(new_parts))

        # advection down_projection:
        if ".down_projection.0-" in k:
            candidate_keys.append(
                k.replace(".down_projection.0-", ".down_projection.body.0-")
            )

        # advection up_projection:
        if ".up_projection.0-" in k:
            candidate_keys.append(
                k.replace(".up_projection.0-", ".up_projection.body.0-")
            )

        # output projection:
        if k.startswith("model.output_proj."):
            parts = k.split(".")

            if len(parts) >= 3 and parts[2].startswith(("0-", "1-")):
                new_parts = parts[:2] + ["body"] + parts[2:]
                candidate_keys.append(".".join(new_parts))

        for new_k in candidate_keys:
            if new_k in new_state and new_state[new_k].shape == v.shape:
                new_state[new_k] = v
                body_mapped.append((k, new_k))
                break

    # -------------------------------------------------------------- #
    # 4. Neutralize noise-conditioning layers
    # -------------------------------------------------------------- #
    for k in list(new_state.keys()):
        if k.endswith("noise_scale.weight"):
            new_state[k] = torch.zeros_like(new_state[k])
            noise_neutralized.append(k)

        elif k.endswith("noise_scale.bias"):
            new_state[k] = torch.ones_like(new_state[k])
            noise_neutralized.append(k)

        elif k.endswith("noise_bias.weight"):
            new_state[k] = torch.zeros_like(new_state[k])
            noise_neutralized.append(k)

        elif k.endswith("noise_bias.bias"):
            new_state[k] = torch.zeros_like(new_state[k])
            noise_neutralized.append(k)

    # -------------------------------------------------------------- #
    # 5. Strict load after building a complete ensemble state_dict
    # -------------------------------------------------------------- #
    litmodel.load_state_dict(new_state, strict=True)

    # -------------------------------------------------------------- #
    # 6. Logging
    # -------------------------------------------------------------- #
    if litmodel.global_rank == 0:
        loaded_old_keys = set(directly_copied)
        loaded_old_keys.update(old_k for old_k, _ in norm_mapped)
        loaded_old_keys.update(old_k for old_k, _ in body_mapped)

        truly_skipped = [
            k for k in det_state.keys()
            if k not in loaded_old_keys
        ]

        suffix_counter = Counter()

        for k in truly_skipped:
            if "GlobalBias.A" in k:
                suffix_counter["GlobalBias.A"] += 1
            elif "GlobalBias.U" in k:
                suffix_counter["GlobalBias.U"] += 1
            elif "GlobalBias.V" in k:
                suffix_counter["GlobalBias.V"] += 1
            else:
                suffix_counter["other"] += 1

        logging.info("=" * 40)
        logging.info("Initialized ensemble model from deterministic checkpoint")
        logging.info(f"Checkpoint: {checkpoint_path}")
        logging.info(f"Directly copied keys: {len(directly_copied)}")
        logging.info(f"Mapped norm keys: {len(norm_mapped)}")
        logging.info(f"Mapped renamed body keys: {len(body_mapped)}")
        logging.info(f"Neutralized noise-conditioning keys: {len(noise_neutralized)}")
        logging.info(f"Truly skipped deterministic keys: {len(truly_skipped)}")
        logging.info(f"Skipped key summary: {dict(suffix_counter)}")
        logging.info("=" * 40)


def maybe_update_backbone_lr_after_catchup(litmodel) -> None:
    """Restore backbone LR after the conditioning catch-up phase.

    During the catch-up phase, backbone parameters stay in the optimizer,
    but their LR is zero. This avoids DDP unused-parameter issues.
    """

    catchup_steps = litmodel.cfg.training.get("conditioning_catchup_steps", 0)

    if not litmodel.ensemble_mode or catchup_steps <= 0:
        return

    if not hasattr(litmodel, "_backbone_lr_restored"):
        litmodel._backbone_lr_restored = False

    if litmodel.global_step >= catchup_steps and not litmodel._backbone_lr_restored:
        opt = litmodel.optimizers()

        for group in opt.param_groups:
            if group.get("name") == "backbone":
                group["lr"] = litmodel.cfg.training.optimizer.lr

        litmodel._backbone_lr_restored = True

        if litmodel.global_rank == 0:
            logging.info("=" * 40)
            logging.info(
                f"Conditioning catch-up finished at global_step={litmodel.global_step}"
            )
            logging.info(
                f"Backbone LR restored to {litmodel.cfg.training.optimizer.lr}"
            )
            logging.info("=" * 40)