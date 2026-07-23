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

## What does not exist yet (planned)

These are planned phases, not features:

| Planned capability | Phase |
|---|---|
| Real WebAssembly execution (Wasmtime-backed: validate, instantiate, invoke, WASI, fuel) | 2 |
| Native SIMD kernels (x86-64 AVX2/FMA, ARM64 NEON) with runtime CPU detection and disassembly verification | 3 |
| Browser WASM SIMD128 kernels with separate scalar/simd bundles and binary instruction verification | 4 |
| Structural WASM pattern analysis (parsed modules, evidence-based confidence) | 5 |
| Safe specialization (host-kernel imports, narrowly-scoped Cranelift JIT) | 6 |
| Reproducible benchmarks with recorded environment and correctness gates | 7 |

## Benchmarks

There are currently **no published benchmark numbers**. Previous README tables
were produced by ad-hoc browser pages and synthetic sleep-based benchmarks and
have been removed. Reproducible benchmarks with recorded methodology,
environment metadata, and correctness verification are Phase 7 scope; published
tables will be generated from checked-in benchmark JSON only.

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
wasm-pack build --target web --features wasm
```

This produces a `pkg/` directory with the WASM binary and JavaScript glue. All
exported kernels are scalar; see [docs/WASM.md](docs/WASM.md).

### CLI

```bash
./target/release/barq-wasm --file module.wasm
```

Currently exits with an error: module execution is not implemented until
Phase 2. The CLI will never report success without executing the module.

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
