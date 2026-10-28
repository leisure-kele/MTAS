# Reproduction guide

## Protocol at a glance

- 26 atomic-action classes at 30 Hz.
- 8-second windows with a 4-second stride for training.
- Complete-sequence evaluation after overlap averaging.
- Decoder selection on validation data only.
- Three paired seeds: `20260824`, `20260825`, and `20260826`.
- H0 reference training: AdamW, learning rate `3e-4`, batch size 16 per GPU, three epochs, and bfloat16 where supported.

The exact scalar contract is stored in `configs/paper/protocol_v3_h0.json`. It documents the paper's experiment specification; it is not a model checkpoint or a substitute for the released data manifest.

## Workflow

### 1. Export or obtain cached episodes

The `atomseg-export-formal500h` entry point converts synchronized recordings to the episode schema. For a distributed feature package, skip this step and use the supplied HDF5 cache and manifest. The released data documentation will provide the exact export command and channel mapping.

### 2. Construct a group-disjoint training release

```bash
atomseg-build-training-release --help
atomseg-prepare-wave1-data --help
```

The release builder validates episode metadata and creates the split and window artifacts. Standardization statistics are estimated from the training split only.

### 3. Train paired models

```bash
atomseg-train-wave1 --help
atomseg-train-crossarch --help
atomseg-train-asformer --help
atomseg-train-pose --help
```

Within each temporal backbone, V and VT are trained as separate models under matched training budgets. The code supports the visual-only, visual-plus-tactile, pose/kinematics, and stage-learning variants described in the paper.

### 4. Evaluate full sequences and interventions

```bash
atomseg-evaluate-robustness --help
```

Window logits are overlap-averaged before decoding. After validation selection, model weights and decoder parameters remain fixed while visual feature blocks or views are made unavailable. The intervention concerns feature/view availability; it is not a natural-occlusion simulator.

## Supplementary fusion controls

`supplementary/aspnet_style/model.py` contains the adapted shared-private architecture. It preserves the shared/private channel partition, MMD alignment, and layer-wise shared bottleneck concept while using this repository's sensor encoders and compact temporal decoder. The supplementary source is intentionally separated because it is not an official reproduction of ASPnet and needs the same release manifest/HDF5 interface as the main pipeline.

## Expected reproducibility boundary

The unit checks can be run without data. End-to-end numerical reproduction needs the data package, frozen feature producer, manifests, and the relevant hardware environment. The repository deliberately does not emulate private test-set access or distribute private test metadata.
