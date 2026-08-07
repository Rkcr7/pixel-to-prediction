#!/usr/bin/env bash
# Full build: Rust tests, WASM, typecheck, production bundle.
#
# The model weights are committed under web/public/model, so a normal build does not
# need MNIST. Run scripts/train.sh only when you want to retrain.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> cargo test"
cargo test --lib

echo "==> wasm-pack build"
(cd crates/nnviz && wasm-pack build --target web --out-dir ../../web/src/wasm --release --no-typescript)

# `npm ci` wipes node_modules first, which fails on Windows while a dev server still
# holds esbuild.exe. Only install when there is nothing installed.
if [ ! -d web/node_modules ]; then
  echo "==> installing web dependencies"
  (cd web && if [ -f package-lock.json ]; then npm ci; else npm install; fi)
else
  echo "==> web dependencies already installed"
fi

echo "==> typecheck"
(cd web && npx tsc --noEmit)

echo "==> vite build"
(cd web && npx vite build)

echo
echo "Built to web/dist. Transfer sizes:"
find web/dist -type f \( -name '*.js' -o -name '*.wasm' -o -name '*.css' -o -name '*.bin' -o -name '*.html' \) \
  -exec sh -c 'printf "  %8s  %s\n" "$(wc -c <"$1")" "${1#web/dist/}"' _ {} \;
