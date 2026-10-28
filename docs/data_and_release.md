# Data and release boundary

The paper studies 44,589 complete bimanual manipulation sequences (380.39 hours). This code repository intentionally contains no recordings, participant imagery, tactile streams, HDF5 caches, feature caches, train/validation/test manifests, trained checkpoints, or private test artifacts.

The planned public release is a 100-hour subset of the multimodal corpus with atomic-action labels and temporal boundaries. Its exact coverage, file checksums, data-use terms, and reproducible evaluation subset will be supplied with the data-release tag rather than inferred from this source tree.

The paper reports results on the full 380.39-hour corpus. Consequently, a public-subset run should not be represented as reproducing every numerical result in the paper unless the corresponding evaluation manifest explicitly establishes that scope.

## Expected training artifacts

The training and evaluation modules consume a release-specific directory containing:

```text
<release-root>/
  summary.json
  train_episodes.jsonl
  val_episodes.jsonl
  test_episodes.jsonl             # only when an evaluation split is distributed
  train_windows.jsonl
  val_windows.jsonl
  test_windows.jsonl
  train_statistics.npz
  <episode-cache>.h5
```

Episode HDF5 files use 60 Hz synchronized source streams and 30 Hz model targets. The public manifest will specify the authoritative field names and checksums. Source code validates required datasets, left/right ordering, target label range, and split membership before training.

## Privacy and access

Do not add raw videos, identifiable participant information, private test manifests, credentials, or full-resolution visual feature caches to a public Git repository. Use the data-release terms to determine whether derivative clips or model checkpoints may be redistributed.
