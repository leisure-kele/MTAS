# When Does Touch Help?

Code for *When Does Touch Help? Multimodal Temporal Action Segmentation in Human Bimanual Manipulation*.

[中文说明](README.zh-CN.md)

## Overview

This repository implements a study of measured tactile sensing in temporal action segmentation of human bimanual manipulation. The task predicts 26 atomic action classes and their temporal transitions from synchronized multimodal observations. A fixed mapping from atomic actions to eight manipulation stages supports a coarser semantic analysis.

The experimental protocol compares visual-only (V) and visual--tactile (VT) models within matched temporal backbones. It evaluates complete sequences after overlap aggregation and applies controlled visual-feature and view-availability interventions with model weights and decoding parameters held fixed. Additional pose and derived-kinematics controls examine the contribution of tactile observations in the presence of motion-state information.

## Repository contents

| Directory | Contents |
| --- | --- |
| `src/atomseg/` | Episode export, data preparation, modality encoders, temporal models, objectives, decoding, and evaluation. |
| `supplementary/aspnet_style/` | Shared-private ASPnet-style adapted fusion control. |
| `configs/paper/` | Fixed H0 training contract and paper analysis specification. |
| `ontologies/` | Definitions of 26 atomic actions and the fixed 26-to-8 stage mapping. |
| `tests/` | Unit and architecture-level tests that do not require the released data. |
| `docs/` | Data-release boundary, required artifacts, and reproduction procedure. |

The main temporal-model suite contains adapted MS-TCN++, ASRF-style, ASFormer-style, and DiffAct-style heads. The M2R2-style and ASPnet-style components are data-interface adaptations for controlled comparison; they are not official reproductions of the respective original methods.

## Installation

```bash
git clone https://github.com/leisure-kele/MTAS.git
cd MTAS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

Feature export additionally requires the packages in the `export` extra and an official local DINOv3 installation:

```bash
pip install -e '.[dev,export]'
```

The visual encoder is frozen in the reported experiments.

## Protocol

- 26 atomic actions, evaluated at 30 Hz.
- Eight-second training windows with a four-second stride.
- Three paired seeds: `20260824`, `20260825`, and `20260826`.
- Validation-only checkpoint selection and decoder calibration.
- Complete-sequence scoring after overlap averaging.
- Fixed model weights and decoding parameters during visual-availability evaluation.

The scalar H0 settings are recorded in [`configs/paper/protocol_v3_h0.json`](configs/paper/protocol_v3_h0.json). The analysis specification is provided in [`configs/paper/analysis_plan_v1.json`](configs/paper/analysis_plan_v1.json).

## Basic checks

```bash
bash scripts/run_unit_checks.sh
python -m atomseg.formal500h_wave1_train --help
python -m atomseg.formal500h_wave4_crossarch_h0 --help
python -m atomseg.formal500h_asformer --help
```

## Reproduction

The end-to-end workflow is:

1. Obtain the distributed data package and manifest.
2. Export or load synchronized episode-HDF5 files.
3. Construct group-disjoint train/validation/test artifacts and training-only normalization statistics.
4. Train paired modality configurations under the fixed protocol.
5. Select weights and decoding settings on validation data.
6. Evaluate full sequences and prescribed visual-availability conditions.

Further details are given in [docs/reproduction.md](docs/reproduction.md). The expected data artifacts and release scope are documented in [docs/data_and_release.md](docs/data_and_release.md).

## Data availability

The paper reports experiments on 44,589 complete sequences totaling 380.39 hours. This repository does not include recordings, participant imagery, tactile streams, feature caches, manifests, checkpoints, or private test artifacts. A 100-hour public subset with multimodal recordings, atomic-action labels, and temporal boundaries will be documented with its data-release version. Results from that subset should not be treated as numerical reproductions of full-corpus results unless the corresponding evaluation manifest explicitly establishes that equivalence.

## License

The code is released under the [MIT License](LICENSE). Data, derived features, and model checkpoints are governed by their respective distribution terms.
