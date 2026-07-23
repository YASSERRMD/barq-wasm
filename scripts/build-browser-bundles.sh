#!/usr/bin/env bash
# Build the two browser bundles:
#   pkg/scalar — compiled with simd128 explicitly disabled
#   pkg/simd   — compiled with simd128 explicitly enabled
#
# JavaScript must feature-detect SIMD support and load the matching bundle
# (see docs/browser/loader.js). Never load the simd bundle unconditionally.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "Building scalar bundle..."
RUSTFLAGS="-C target-feature=-simd128" \
  wasm-pack build --target web --out-dir pkg/scalar --no-default-features --features browser

echo "Building simd bundle..."
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --target web --out-dir pkg/simd --no-default-features --features browser

echo "Bundles built: pkg/scalar, pkg/simd"
