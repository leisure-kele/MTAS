# When Does Touch Help?

Code accompanying **“When Does Touch Help? Multimodal Temporal Action Segmentation in Human Bimanual Manipulation.”**

This repository contains the paper-facing implementation for temporal action segmentation of multimodal human bimanual manipulation. It supports the 26-class atomic-action task, a fixed 26-to-8 manipulation-stage mapping, complete-sequence evaluation, paired visual-availability interventions, and the adapted temporal backbones used in the paper.

## What is included

- `src/atomseg/`: data preparation, modality encoders, loss functions, complete-sequence decoding, evaluation, and the four adapted temporal backbones (MS-TCN++, ASRF-style, ASFormer-style, and DiffAct-style).
- `supplementary/aspnet_style/`: the shared-private ASPnet-style fusion architecture used as a supplementary control. It is an adaptation for this data interface, not an official reproduction of ASPnet.
- `configs/paper/`: the fixed H0 training contract and analysis plan used to specify the paper protocol.
- `ontologies/`: the 26 atomic actions and the fixed eight-stage mapping.
- `tests/`: unit and architecture-level checks that do not require the released dataset.
- `docs/`: data-release boundaries, expected artifacts, and a reproduction guide.

No participant data, video, tactile recordings, precomputed features, checkpoints, cached HDF5 files, private test manifests, server paths, or experiment logs are included.

## Installation

```bash
git clone <YOUR-REPOSITORY-URL>
cd when-does-touch-help
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

For DINOv3 feature extraction, install the DINOv3 implementation and weights through their official release, then point the feature-export configuration to that local installation. The visual backbone is frozen in all paper experiments.

## Quick checks

```bash
pytest -q tests/test_model.py tests/test_formal500h_clock.py tests/test_formal500h_protocol.py
python -m atomseg.formal500h_wave1_train --help
python -m atomseg.formal500h_wave4_crossarch_h0 --help
python -m atomseg.formal500h_asformer --help
```

## Reproduction outline

1. Obtain the released data package and its accompanying manifest.
2. Export synchronized recordings into the documented episode-HDF5 schema.
3. Build the group-disjoint training release and 30 Hz windows.
4. Train matched V and VT variants with the fixed H0 contract.
5. Select checkpoints and decoder settings on validation data only.
6. Run complete-sequence evaluation and the specified visual-feature/view interventions with fixed weights and decoder settings.

Detailed commands and required artifacts are in [docs/reproduction.md](docs/reproduction.md). The published 100-hour subset and its label files will be documented in the repository release when distribution is enabled; the paper's reported results use the full 380.39-hour corpus.

## Scope and citation

This repository is a research release. The four temporal backbones and the M2R2-style/ASPnet-style controls are adaptations to a common multimodal interface; they must not be interpreted as official reproductions or cross-dataset rankings of the original methods.

The code is released under the [MIT License](LICENSE). The dataset, videos, feature caches, and checkpoints are not included and remain subject to their separate access and distribution terms.
