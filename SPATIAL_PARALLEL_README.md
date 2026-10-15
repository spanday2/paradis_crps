# CRPS + PARADIS spatial parallel merge (unvalidated reference integration)

This is a source-level integration constructed by combining the supplied CRPS tree and the deterministic spatial-parallel tree. It has NOT been run on distributed CUDA GPUs or an ERA5 dataset. Run correctness and numerical-gradient checks before production use.

Set `compute.num_devices` and `compute.spatial_parallel.enabled: true`, `compute.spatial_parallel.size: 2` (or 4). Use Lightning DDP. Keep `compute.compile: false` and `gradient_checkpointing: false` for initial debugging.

Retains noise embedding and conditional normalization, two-member almost-fair CRPS, spectral CRPS, ensemble forecast output layout and CRPS checkpoint loading. The latitude-band CRPS uses global latitude weights plus spatial-size/local-height scaling. The spectral term gathers a full field using an autograd-aware collective. Writers run only on group leaders.

**Not verified:** CUDA/NCCL multi-process run, distributed autograd, training stability, multi-node writes, target data in memory, quantitative agreement with spatial_size=1. Compare single-GPU vs spatial outputs and gradients before use.
