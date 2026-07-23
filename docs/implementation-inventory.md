# Implementation Inventory — Truth Baseline (Phase 1)

This document records, file by file, what the repository actually implemented at the
start of Phase 1, and what was misrepresented. It is the basis for
`implementation-status.json` and for the phased reimplementation plan.

Legend:

- **implemented** — real, executable, tested behavior
- **partial** — some real behavior exists but claims exceed it
- **absent** — advertised or stubbed, but no real behavior exists

## src/runtime

| File | Finding | Status |
|---|---|---|
| `mod.rs` | Module list only. | absent |
| `memory_manager.rs` | `MemoryPool` held only `_capacity`; `allocate()` was `todo!()`. | absent |
| `profiling.rs` | `ExecutionProfiler` held `()`; `start()` was `todo!()`. | absent |
| `adaptive_recompilation.rs` | `AdaptiveCompiler` held `()`; `recompile()` was `todo!()`. | absent |

No WebAssembly module was ever loaded, validated, instantiated, or executed by this
crate. **Action:** placeholder structs deleted; a real Wasmtime-backed runtime is
Phase 2 scope.

## src/executor.rs

`BarqRuntime::new()` constructed an empty tuple and returned `Ok`, which let the CLI
print "runtime initialized successfully" without any runtime existing. `run()` was
`todo!()`. **Action:** constructor now returns a typed
`BarqError::UnsupportedFeature` error until a real runtime exists.

## src/codegen

| File | Finding | Status |
|---|---|---|
| `cranelift_backend.rs` | `compile()` returned `machine_code: vec![0x90, 0x90]` (two x86 NOPs) for every input. No Cranelift API was used despite the name. | absent |
| `vector_codegen.rs` | CPU feature "detection" hard-coded AVX2/SSE4.2 as present, including on ARM. Emitters pushed hand-written byte arrays that were never executed. | absent |
| `ai_codegen.rs` | Emitted fixed byte arrays labeled as VNNI/AVX instructions; never executed. | absent |
| `compression_codegen.rs` | Pushed strings like `"unroll_loop_4"` into a fake IR vector. | absent |
| `database_codegen.rs` | Same string-pushing approach. | absent |

**Action:** the entire module tree is deleted. Real code generation (Cranelift JIT
for narrow signatures) is Phase 6 scope and must never emit unverified byte arrays.

## src/analyzer and src/patterns

Pattern "detection" counted opcode bytes in caller-supplied arrays that were not
WebAssembly modules, then returned hard-coded confidence values (e.g. `0.8` when 25
multiply bytes were present). Claims of detecting "LZ4", "attention", "MongoDB"
had no structural basis. **Action:** deleted. Structural analysis over parsed WASM
(wasmparser-based) is Phase 5 scope.

## src/syscalls

All three modules (`compression`, `database`, `vector`) were `todo!()` bodies.
The README's "Adaptive Syscall Mapping" feature never existed. **Action:** deleted.

## src/wasm_bindings.rs

The compute kernels here are real, executable scalar Rust. The misrepresentation
was in naming and documentation:

| Function (old name) | Reality | Action |
|---|---|---|
| `dot_product_simd` | 16-wide **unrolled scalar** with pointer arithmetic; no `v128`/intrinsics. | renamed `dot_product_unrolled_scalar` |
| `quantize_int8_simd` | Doc claimed "native WASM SIMD ... v128 instructions"; body is unrolled scalar. | renamed `quantize_int8_unrolled_scalar`; false claims removed |
| `vector_norm_simd` | Unrolled scalar. | renamed `vector_norm_unrolled_scalar` |
| `cosine_similarity_simd` | Composition of the above. | renamed `cosine_similarity_unrolled_scalar` |
| `lz4_compress_optimized` | Returns the input **verbatim** for buffers < 128 KiB; larger buffers use a hash-based LZ4-like block emitter with no decompressor and no format conformance test. | doc rewritten to state this; status `partial` |
| `matrix_multiply_tiled`, `conv2d_optimized`, elementwise/statistics ops | Real scalar implementations; performance claims in doc comments were unverified. | perf claims removed from docs |

No function in this crate executes SIMD instructions explicitly. Whether the
compiler auto-vectorizes any loop has never been verified by disassembly, so no
such claim is made anywhere.

## benches

All four benchmark files (`compression_bench.rs`, `vector_bench.rs`,
`database_bench.rs`, `ai_bench.rs`) measured `thread::sleep` calls with hard-coded
nanosecond durations chosen to fabricate speedup ratios. `temp_bench/` HTML pages
compared against those numbers. **Action:** all deleted. Real Criterion and
browser benchmarks are Phase 7 scope.

## tests

`tests/integration_tests.rs` asserted that the fake backends returned `Ok` and that
fake IR vectors contained marker strings — i.e., it locked in the simulated
behavior. **Action:** replaced with tests that (a) verify the renamed scalar
kernels compute correct results, and (b) verify unavailable functionality returns
typed errors instead of fake success.

## CLI (src/main.rs)

Printed "Barq-WASM runtime initialized successfully." after constructing an empty
struct, and ignored the `--file` argument entirely. **Action:** now exits non-zero
with a typed error explaining that module execution is not yet implemented.

## Browser package (pkg/) and README

`pkg/` is a generated wasm-pack artifact exposing the falsely-named `*_simd`
exports; it will be regenerated after the kernels are renamed (and properly in
Phase 4 with real SIMD128 bundles). README claims about pattern detection,
specialized JIT tiers, and syscall bypass describe code that never executed and
are re-labeled as planned work until reimplemented and verified.
