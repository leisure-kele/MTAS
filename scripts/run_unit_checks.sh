#!/usr/bin/env bash
set -euo pipefail

python scripts/verify_release.py

pytest -q \
  tests/test_model.py \
  tests/test_formal500h_clock.py \
  tests/test_formal500h_protocol.py \
  tests/test_formal500h_asformer.py \
  tests/test_formal500h_wave4_crossarch.py
