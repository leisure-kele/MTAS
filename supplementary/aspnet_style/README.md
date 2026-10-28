# ASPnet-style adapted fusion control

`model.py` implements the supplementary shared-private fusion control used in the paper. Each enabled modality is projected to a 64-dimensional stream and partitioned into 32-dimensional private and shared components. Shared components are aligned with multi-kernel MMD on co-observed time steps and exchange information through a layer-wise shared bottleneck. The resulting representation is decoded by a compact multi-stage temporal head.

This is an adaptation for the common V/T/P/K data interface in this repository. It reuses the project sensor encoders and is **not** an official implementation or reproduction of ASPnet. It requires the same episode-HDF5 and release-manifest interface as the main pipeline.

The auxiliary files provide synthetic fixtures, pose-only reading helpers, and group-level statistical utilities. They do not contain experiment results, test manifests, paths, or checkpoints.
