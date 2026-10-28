# Code map

| Area | Main files | Purpose |
| --- | --- | --- |
| Episode export | `formal500h_export.py`, `exporter.py`, `mcap_io.py` | Synchronize recordings and create the episode-HDF5 representation. |
| Data/window preparation | `formal500h_training_release.py`, `formal500h_wave1_data.py` | Validate manifests, create group-disjoint splits, derive 30 Hz targets, and estimate train-only statistics. |
| Modality representation | `model.py`, `baselines.py`, `formal500h_runtime.py` | Encode multiview vision, tactile values/differences, joint pose, and derived kinematics. |
| Temporal backbones | `formal500h_wave4_crossarch.py`, `formal500h_asformer.py`, `formal500h_wave1_train.py` | Adapt MS-TCN++, ASRF-style, DiffAct-style, and ASFormer-style heads to the common interface. |
| Objectives and decoding | `losses.py`, `inference.py`, `formal500h_evaluation.py` | Optimize action/boundary objectives, stitch window predictions, decode segments, and compute metrics. |
| Robustness and controls | `formal500h_wave5_robustness.py`, `formal500h_pose_train.py`, `formal500h_phase_train.py` | Evaluate visual-availability conditions, pose/kinematics controls, and the fixed 26-to-8 stage view. |
| Supplementary fusion | `supplementary/aspnet_style/model.py` | Adapted shared-private fusion control. |

The repository excludes deployment wrappers, server schedulers, test authorizations, experiment logs, private test paths, and historical pre-paper pipelines.
