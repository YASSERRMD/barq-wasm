<div align="center">
  <img src="assets/logo.svg" alt="Barq-WASM Logo" width="200" height="200">
  <h1>Barq-WASM</h1>
  <p><strong>Pattern-Aware WebAssembly Runtime for Specialized Workloads (in rebuild)</strong></p>

  [![CI](https://github.com/YASSERRMD/barq-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/YASSERRMD/barq-wasm/actions)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
</div>

## Project status: truth baseline

This repository previously contained prototype code whose names and documentation
claimed more than was implemented (functions named `*_simd` that were scalar,
a "Cranelift backend" that emitted NOP bytes, pattern "detection" that counted
opcode bytes in non-WASM arrays, and benchmarks that timed `thread::sleep`).

All of that has been removed or renamed. The authoritative record of what exists
now is:

- [`implementation-status.json`](implementation-status.json) — machine-readable
  capability status (implemented / partial / absent), with tests and verification
  per capability.
- [`docs/implementation-inventory.md`](docs/implementation-inventory.md) — the
  file-by-file audit that produced the baseline.

No capability is claimed below unless it is implemented, tested, and verified.

## What works today (verified)

**Real WebAssembly execution** (`src/runtime`, Wasmtime-backed):

- Module validation, loading (`.wasm` and `.wat`), instantiation, typed and
  dynamically-typed invocation with concrete-value tests.
- Linear memory read/write from host and guest, with typed out-of-bounds
  errors.
- WASI preview1 (optionally with captured stdout/stderr), fuel limits with
  consumption reporting, wall-clock deadlines via epoch interruption, and
  guest memory limits.
- Traps (unreachable, division by zero, out-of-fuel, interrupt) map to typed
  `BarqError` variants.
- CLI: `barq-wasm validate|inspect|run|benchmark`. Example:
  `barq-wasm run fixtures/add.wasm --invoke add --arg-i32 20 --arg-i32 22`
  prints `42`. The `benchmark` subcommand refuses to publish timings if
  results are non-deterministic across iterations.

Verified by 16 runtime integration tests over real WAT fixtures plus 9
end-to-end CLI tests (`tests/runtime_integration.rs`, `tests/cli.rs`).

**Native SIMD kernels** (`src/kernels`): explicit AVX2/AVX2+FMA (x86-64) and
NEON (ARM64) implementations of dot product, L2 norm, cosine similarity, INT8
quantize/dequantize, and matrix multiply, with scalar references as ground
truth.

- Runtime CPU detection (`is_x86_feature_detected!` /
  `is_aarch64_feature_detected!`, cached once). Safe wrappers refuse to run on
  unsupported CPUs with typed errors. `BARQ_FORCE_KERNEL=scalar|avx2|avx2-fma|neon`
  forces a backend for testing; forcing an unsupported one is an error, never
  an illegal instruction.
- Auto-dispatch entry points (`dot_product`, `quantize_i8`, ...) report which
  backend actually ran (`KernelExecution { value, backend }`). conv2d is
  scalar-only and reports exactly that.
- Quantization semantics are defined (IEEE round-to-nearest-even, saturating,
  NaN→0) and the differential tests require **bit-identical** output across
  scalar, AVX2, and NEON.
- `scripts/verify-native-simd.sh` disassembles the release build and fails
  unless real SIMD instructions are present (`vmulps`/`vfmadd`/`ymm` on
  x86-64; `fmla`/`.4s` on ARM64). Wired into CI.
- Verified by differential tests over boundary lengths (empty, 1, below/at/
  above vector width, 10k+), sign/zero/NaN/infinity policies, misaligned
  slices, and property-based randomized tests.

**Browser WASM SIMD128 kernels** (`src/kernels/wasm32/simd128.rs`): explicit
`v128`/`f32x4` implementations of dot product, L2 norm, cosine similarity,
INT8 quantize/dequantize (bit-identical to the scalar policy), and
elementwise ops.

- Shipped as two bundles: `pkg/scalar` (zero SIMD instructions, verified) and
  `pkg/simd` (`v128.load`/`f32x4.mul`/`f32x4.add`/`i32x4.trunc_sat_f32x4_s`
  verified present via the wasmparser-based `wasm-inspect` tool).
- `docs/browser/loader.js` feature-detects SIMD before loading the simd
  bundle and cross-checks the bundle's own `simd128_enabled()` report.
- Verified by headless-browser tests (Chrome locally and in CI, Firefox in
  CI): differential vs scalar, bit-exact quantization including
  NaN/infinity/ties, repeated-invocation stability, and 500k-element
  transfers.

**Structural WASM pattern analysis** (`src/analyzer`): parses real modules
with `wasmparser` and symbolically executes each function body into loop
facts (load/store address expressions, accumulators, induction variables).

- Detects dot-product, matrix-multiply, and quantization candidates from
  mandatory structural evidence — e.g. a dot product requires a
  multiply-accumulate reduction over two loads with distinct base pointers;
  sum-of-squares and elementwise loops are structurally rejected.
- Confidence is satisfied-requirements/total and every candidate carries its
  evidence list (`barq-wasm analyze module.wasm` prints them). Tests assert
  it is never a constant.
- Verified on a labeled WAT corpus (indexed, pointer-bump, and noisy shapes;
  near-misses; unrelated code) with precision and recall asserted at 1.0;
  fabricated non-WASM byte arrays are rejected with typed errors.
- Detection only: nothing is substituted or accelerated by the analyzer
  (that is Phase 6, with differential safety tests).

**Safe specialization** (`src/runtime/barq_abi.rs`, `src/jit/`):

- *Host-kernel ABI*: guests that import `barq.dot_product_f32`,
  `barq.l2_norm_f32`, `barq.cosine_similarity_f32`, or `barq.quantize_i8` get
  the crate's verified native kernels (AVX2/FMA, NEON, or scalar via the same
  dispatch). Opt-in only — nothing is silently rewritten. Bounds and
  alignment are validated; violations trap as typed errors. Differentially
  tested: host result vs a pure-guest scalar loop vs the native reference on
  identical guest memory; quantization output is bit-exact.
- *Cranelift JIT* (`jit-specialization` feature): compiles complete IR
  functions for narrow signatures (f32 dot product with a real loop; i64
  add), resolves callable pointers, executes through length-validated typed
  wrappers, and frees executable memory safely on drop. Differentially
  tested against scalar references including 200 randomized lengths. There
  is deliberately no test that merely checks machine code is non-empty.
- Automatic substitution of analyzer-detected patterns is **not** performed;
  `barq-wasm analyze` only points at the opt-in ABI.

**Scalar compute kernels** (`src/wasm_bindings.rs`), exposed to JavaScript via
`wasm-bindgen` and usable natively:

- Dot product, L2 norm, cosine similarity — naive scalar baselines plus
  manually unrolled scalar variants (`*_unrolled_scalar`). The unrolled variants
  use multiple independent accumulators for instruction-level parallelism; they
  do **not** execute SIMD instructions.
- Cache-tiled matrix multiplication (`matrix_multiply_tiled`) and a naive
  reference (`matrix_multiply_scalar`).
- INT8 quantization/dequantization, Conv2D with an unrolled 3x3 fast path,
  pooling, softmax/sigmoid/relu, elementwise vector ops, statistics.

Correctness is enforced by differential tests against naive scalar references
(`tests/truth_baseline.rs`).

**Typed failure for everything else.** The native runtime entry point returns
`BarqError::UnsupportedFeature`; the CLI exits non-zero. Nothing prints success
without executing real work.

## What does not exist yet

- Automatic substitution of analyzer-detected patterns (opt-in ABI only).
- SIMD conv2d and AVX-512 kernels (detected but unimplemented; conv2d reports
  scalar truthfully).
- Syscall bypass / "adaptive syscall mapping" — never existed, still absent.

## Benchmarks

Reproducible benchmarks live in three places, all correctness-gated (a result
is verified against the scalar reference **before** any timing; an incorrect
implementation gets no timing at all):

- `cargo run --release --features bench-tool --bin barq-bench` — native
  harness with warm-up, median/p90/p95/stddev, size sweep (15 → 1 000 000;
  matrices 16 → 256), and embedded environment metadata (OS, CPU, detected
  features, selected backend, git commit). Exits non-zero on any incorrect
  result.
- `cargo bench` — Criterion benchmarks over the same kernels with the same
  correctness gates.
- `docs/browser/benchmark.html` — browser harness comparing a JS baseline
  against wasm scalar / unrolled / SIMD128 with identical warm-up and sample
  counts on identical data.

Checked-in raw results: [`benchmarks/results/`](benchmarks/results/)
(currently one Apple M4 / macOS run; every record carries `correct: true`
and full stats). `tests/bench_integrity.rs` validates every checked-in file
in CI and rejects sleeps or fabricated durations in benchmark sources. No
speedup claims are published in prose — read the JSON, which names the
hardware it came from.

## Installation

### Prerequisites

- Rust (latest stable toolchain)
- wasm-pack (for building the browser package): `cargo install wasm-pack`

### Building from source

```bash
git clone https://github.com/YASSERRMD/barq-wasm.git
cd barq-wasm
cargo build --release
cargo test
```

### Building for the browser

```bash
./scripts/build-browser-bundles.sh   # pkg/scalar + pkg/simd
./scripts/verify-wasm-simd.sh        # verify instruction content
```

See [docs/WASM.md](docs/WASM.md) for the API and
[docs/browser/loader.js](docs/browser/loader.js) for feature-detected loading.

### CLI

```bash
./target/release/barq-wasm validate module.wasm
./target/release/barq-wasm inspect module.wasm
./target/release/barq-wasm run module.wasm --invoke add --arg-i32 20 --arg-i32 22
./target/release/barq-wasm benchmark module.wasm --invoke f --iterations 100
```

Optional execution guards: `--fuel N`, `--timeout-ms N`, `--max-memory BYTES`,
`--no-wasi`. The CLI exits non-zero on any failure and never reports success
without executing the module.

### Verifying everything

```bash
cargo xtask verify-all                 # every layer this host can run
cargo xtask verify-all --with-browser  # + headless Chrome tests
```

Produces `verification-report.json` and `verification-report.md`. Every step
ends PASS, FAIL, or NOT RUN with the reason — NOT RUN is never shown as PASS,
and any FAIL exits non-zero. CI additionally runs the truthfulness gate
(`scripts/check-truthfulness.sh`), a Linux/macOS/Windows test matrix, release
SIMD verification on both architectures, browser tests in Chrome and Firefox,
an MSRV check, and non-blocking cargo-audit and AddressSanitizer jobs.

## Contributing

Contributions are welcome. Ground rules for this codebase:

1. No function may claim SIMD, JIT, or acceleration it does not verifiably
   perform.
2. Every optimized implementation needs a differential test against a scalar
   reference.
3. Benchmarks must execute real work and record their environment; no
   synthetic timings.
4. `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
   and `cargo test` must pass.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
