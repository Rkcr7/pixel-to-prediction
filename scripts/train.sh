#!/usr/bin/env bash
# Retrain the model from scratch and export the web assets.
#
# Takes about two minutes on eight cores. Refuses to export below the accuracy gate, so
# a bad run cannot silently ship.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/fetch-mnist.sh data

cargo run --release --bin train -- \
  --data data \
  --out web/public/model \
  --epochs "${EPOCHS:-32}" \
  --batch "${BATCH:-64}" \
  --lr "${LR:-2e-3}" \
  --seed "${SEED:-1234}" \
  --gate "${GATE:-0.993}"
