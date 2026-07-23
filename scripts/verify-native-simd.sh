#!/usr/bin/env bash
# Verify that the native SIMD kernels actually contain SIMD instructions.
#
# Builds the library with --emit=asm and asserts that:
#   - the expected kernel symbols are present, and
#   - their architecture's SIMD instructions appear in the assembly.
#
# This guards against silently shipping scalar code under a SIMD name.
# Usage: scripts/verify-native-simd.sh [--target <triple>]

set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=""
if [[ "${1:-}" == "--target" ]]; then
  TARGET="$2"
fi

TARGET_FLAG=()
ASM_DIR="target/release/deps"
if [[ -n "$TARGET" ]]; then
  TARGET_FLAG=(--target "$TARGET")
  ASM_DIR="target/$TARGET/release/deps"
fi

rm -f "$ASM_DIR"/barq_wasm*.s
cargo rustc --release --lib ${TARGET_FLAG[@]+"${TARGET_FLAG[@]}"} -- --emit=asm >/dev/null

ASM_FILE=$(ls -t "$ASM_DIR"/barq_wasm*.s 2>/dev/null | head -1)
if [[ -z "$ASM_FILE" ]]; then
  echo "FAIL: no assembly file produced in $ASM_DIR" >&2
  exit 1
fi
echo "Inspecting: $ASM_FILE"

ARCH=$(rustc ${TARGET:+--target "$TARGET"} --print cfg 2>/dev/null | sed -n 's/^target_arch="\(.*\)"$/\1/p')
echo "Target arch: $ARCH"

fail=0
require_symbol() {
  if grep -q "$1" "$ASM_FILE"; then
    echo "  symbol present: $1"
  else
    echo "FAIL: expected symbol matching '$1' not found" >&2
    fail=1
  fi
}
require_insn() {
  if grep -qE "$1" "$ASM_FILE"; then
    echo "  instruction present: $1"
  else
    echo "FAIL: expected instruction matching '$1' not found" >&2
    fail=1
  fi
}

case "$ARCH" in
  x86_64)
    require_symbol 'kernels.*x86.*avx2.*dot_product_f32'
    require_symbol 'kernels.*x86.*avx2.*quantize_f32_to_i8'
    require_symbol 'kernels.*x86.*avx2.*matrix_multiply_f32_fma'
    require_insn '\bvmulps\b'
    require_insn '\bvaddps\b'
    require_insn 'vfmadd[0-9]*ps'
    require_insn '%ymm[0-9]'
    ;;
  aarch64)
    require_symbol 'kernels.*arm.*neon.*dot_product_f32'
    require_symbol 'kernels.*arm.*neon.*quantize_f32_to_i8'
    require_symbol 'kernels.*arm.*neon.*matrix_multiply_f32'
    require_insn '\bfmla\b'
    require_insn '\bld1\b|ldr[[:space:]]+q[0-9]|ldp[[:space:]]+q[0-9]'
    # 128-bit 4x f32 arrangement: GNU syntax "v0.4s", Apple syntax "fmla.4s v0"
    require_insn 'v[0-9]+\.4s|fmla\.4s'
    ;;
  *)
    echo "FAIL: unsupported arch '$ARCH' for SIMD verification" >&2
    exit 1
    ;;
esac

if [[ $fail -ne 0 ]]; then
  echo "Native SIMD verification FAILED" >&2
  exit 1
fi
echo "Native SIMD verification PASSED ($ARCH)"
