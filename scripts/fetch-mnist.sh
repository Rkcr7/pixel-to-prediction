#!/usr/bin/env bash
# Downloads MNIST into ./data as raw idx files.
#
# yann.lecun.com returns 403 these days, so we use the PyTorch-maintained S3 mirror with a
# GitHub mirror as a fallback.
set -euo pipefail

DIR="${1:-data}"
mkdir -p "$DIR"

PRIMARY="https://ossci-datasets.s3.amazonaws.com/mnist"
FALLBACK="https://raw.githubusercontent.com/fgnt/mnist/master"

FILES=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)

for f in "${FILES[@]}"; do
  if [ -f "$DIR/$f" ]; then
    echo "have $f"
    continue
  fi
  gz="$DIR/$f.gz"
  echo "fetching $f"
  if ! curl -fsSL "$PRIMARY/$f.gz" -o "$gz"; then
    echo "  primary mirror failed, trying fallback"
    curl -fsSL "$FALLBACK/$f.gz" -o "$gz"
  fi
  gzip -df "$gz"
done

echo
ls -la "$DIR"
