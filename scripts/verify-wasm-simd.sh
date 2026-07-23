#!/usr/bin/env bash
# Verify the browser bundles' instruction content and exports:
#   - the simd bundle MUST contain v128.load / f32x4.mul / f32x4.add /
#     i32x4.trunc_sat_f32x4_s and export the *_simd128 functions;
#   - the scalar bundle MUST contain no SIMD instructions and MUST NOT
#     export any *_simd128 function;
#   - both bundles must export the scalar API.
#
# Builds the bundles first unless SKIP_BUILD=1.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  ./scripts/build-browser-bundles.sh
fi

INSPECT=(cargo run -q --features inspect --bin wasm-inspect --)

echo "== scalar bundle =="
"${INSPECT[@]}" pkg/scalar/barq_wasm_bg.wasm \
  --expect-no-simd \
  --require-export dot_product_scalar \
  --require-export quantize_int8_scalar \
  --require-export simd128_enabled \
  --forbid-export dot_product_simd128 \
  --forbid-export quantize_int8_simd128

echo "== simd bundle =="
"${INSPECT[@]}" pkg/simd/barq_wasm_bg.wasm \
  --expect-simd \
  --require-op V128Load \
  --require-op F32x4Mul \
  --require-op F32x4Add \
  --require-op I32x4TruncSatF32x4S \
  --require-export dot_product_simd128 \
  --require-export vector_norm_simd128 \
  --require-export cosine_similarity_simd128 \
  --require-export quantize_int8_simd128 \
  --require-export dequantize_int8_simd128 \
  --require-export dot_product_scalar \
  --require-export simd128_enabled

echo "WASM SIMD verification PASSED"
