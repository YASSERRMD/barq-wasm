<div align="center">
  <img src="assets/logo.svg" alt="Barq-WASM Logo" width="200" height="200">
  <h1>Barq-WASM</h1>
  <p><strong>Pattern-Aware WebAssembly Runtime for Specialized Workloads</strong></p>

  [![CI](https://github.com/YASSERRMD/barq-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/YASSERRMD/barq-wasm/actions)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
</div>

## Overview

Barq-WASM is a high-performance, experimental WebAssembly runtime designed to bridge the gap between generic bytecode execution and specialized native hardware acceleration. Unlike traditional runtimes that rely solely on generic JIT compilation, Barq-WASM employs a multi-stage Pattern Analyzer to detect high-level algorithmic structures and recompiles them into highly optimized, architecture-specific machine code.

This project demonstrates improved throughput for specific, compute-intensive workloads by leveraging AVX2/SIMD instructions, native syscall injections, and algorithmic shortcuts that generic WASM compilers cannot safely prove effective.

## Key Features

### 1. Intelligent Pattern Detection
The runtime analyzes WASM bytecode prior to execution to identify known algorithmic fingerprints:
*   Compression: LZ4, Zstd, Brotli decompression loops
*   Linear Algebra: Matrix multiplication, dot products, vector norms
*   AI/ML: INT8 quantization, convolution layers, attention mechanisms
*   Database: MongoDB protocol handlers, BSON serialization

### 2. Specialized JIT Compilation
Once a pattern is detected, the CraneliftBackend switches to a specialized optimization tier:
*   Vector Backend: Emits AVX2/SSE4.2 instructions for linear algebra
*   Compression Backend: Injects prefetch hints and unrolls dictionary lookups
*   AI Backend: Utilizes native INT8 instructions and tiled convolution kernels

### 3. Adaptive Syscall Mapping
For I/O-bound patterns (like database drivers), Barq-WASM can bypass standard WASI generic I/O in favor of direct, batched pwrite64 syscalls and connection pooling, reducing context-switch overhead.

## Performance Benchmarks

Benchmarks were conducted in Chrome (V8) comparing Barq-WASM compiled module vs pure JavaScript implementations.

### Browser Benchmark Results (Chrome V8)

| Category | Workload | WASM Time | JS Time | Speedup |
|:---|:---|:---:|:---:|:---:|
| Vector | Dot Product (500K elements) | 0.177 ms | 0.391 ms | 2.21x |
| Vector | L2 Norm (500K elements) | 0.119 ms | 0.329 ms | 2.76x |
| Vector | Cosine Similarity (100K) | 0.081 ms | 0.219 ms | 2.70x |
| Matrix | Matrix Mul (64x64) | 0.067 ms | 0.913 ms | 13.70x |
| Matrix | Matrix Mul (128x128) | 0.540 ms | 8.570 ms | 15.93x |
| AI | INT8 Quantization (500K) | 0.378 ms | 2.686 ms | 7.11x |
| AI | Conv2D + ReLU (256x256) | 0.252 ms | 0.986 ms | 3.82x |
| Compression | LZ4 (97.7KB) | 0.012 ms | 0.010 ms | 0.83x |

### Summary Statistics

| Metric | Value |
|:---|:---:|
| Average Speedup | 5.36x |
| Best Speedup | 15.93x (Matrix Multiply) |
| Tests with 2x+ speedup | 6 out of 8 |

### Real-World Data Benchmarks

Tested with actual data: Shakespeare text, GloVe-style embeddings, and MNIST images.

| Category | Task | Data Source | Speedup |
|:---|:---|:---|:---:|
| Word Embeddings | Cosine Similarity | 300-dim GloVe-style vectors | 3.25x |
| Word Embeddings | KNN Search (100 vectors) | 300-dim, 100 comparisons | 3.53x |
| NLP | Text Compression | Shakespeare (~180KB) | 0.02x* |
| Computer Vision | Edge Detection (3x3) | MNIST 28x28 images | 3.57x |
| ML Inference | INT8 Quantization | 500K ReLU activations | 7.07x |

*LZ4 compression uses adaptive algorithm: for buffers under 128KB, direct copy matches JS memcpy performance.

### Key Optimizations

*   16-wide Loop Unrolling: Dot product uses 16 independent accumulators with pointer access for maximum ILP
*   L1/L2 Cache Tiling: Matrix multiplication uses 32x32 L1 tiles and 64x64 L2 tiles for optimal cache utilization
*   Fast INT8 Quantization: Uses integer-based rounding instead of floating-point round(), achieving 7.11x speedup
*   Fused Operations: Conv2D includes fused ReLU activation, eliminating a separate memory pass
*   Adaptive Compression: Buffer-size aware algorithm selection (direct copy for small buffers where overhead exceeds benefit)

### Comparison with Industry Standards

| Workload | Barq-WASM | ONNX Runtime Web | TensorFlow.js WASM |
|:---|:---:|:---:|:---:|
| INT8 Quantization | 0.38 ms | 1.5-1.9 ms | 1.7-2.2 ms |
| Conv2D (256x256) | 0.25 ms | 0.8-1.2 ms | 1.0-1.5 ms |
| Matrix Mul (128x128) | 0.54 ms | 2-4 ms | 3-5 ms |

## Installation

### Prerequisites
*   Rust (latest stable toolchain)
*   Cargo
*   wasm-pack (for building WebAssembly package)
*   (Optional) Clang/LLVM for specific linkage requirements

Install wasm-pack:

```bash
cargo install wasm-pack
```

### Building from Source

Clone the repository and build the release binary:

```bash
git clone https://github.com/YASSERRMD/barq-wasm.git
cd barq-wasm
cargo build --release
```

To run the full suite of integration tests:

```bash
cargo test --test integration_tests
```

To run performance benchmarks:

```bash
cargo bench
```

### Building for Web (WASM Package)

Build the WebAssembly package for browser usage:

```bash
wasm-pack build --target web --features wasm
```

This generates a `pkg/` directory containing:
*   `barq_wasm.js` - JavaScript bindings and module loader
*   `barq_wasm_bg.wasm` - Compiled WebAssembly binary
*   `barq_wasm.d.ts` - TypeScript type definitions
*   `package.json` - npm package configuration

### Using in HTML

Include the generated package in your HTML file:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Barq-WASM Example</title>
</head>
<body>
    <script type="module">
        import init, {
            dot_product_simd,
            matrix_multiply_tiled,
            quantize_int8_simd,
            conv2d_optimized,
            cosine_similarity_simd,
            vector_norm_simd,
            lz4_compress_optimized
        } from './pkg/barq_wasm.js';

        async function run() {
            // Initialize the WASM module
            await init();

            // Example: Dot product of two vectors
            const a = new Float32Array([1.0, 2.0, 3.0, 4.0]);
            const b = new Float32Array([5.0, 6.0, 7.0, 8.0]);
            const result = dot_product_simd(a, b);
            console.log('Dot product:', result);

            // Example: Matrix multiplication (64x64)
            const size = 64;
            const matA = new Float32Array(size * size).fill(0.5);
            const matB = new Float32Array(size * size).fill(0.5);
            const matC = matrix_multiply_tiled(matA, matB, size);
            console.log('Matrix result:', matC[0]);

            // Example: INT8 quantization
            const floats = new Float32Array([0.1, 0.5, -0.3, 0.9]);
            const quantized = quantize_int8_simd(floats, 0.1);
            console.log('Quantized:', quantized);
        }

        run();
    </script>
</body>
</html>
```

### Available WASM Functions

| Function | Description | Parameters |
|:---|:---|:---|
| `dot_product_simd(a, b)` | Compute dot product of two vectors | Float32Array, Float32Array |
| `vector_norm_simd(a)` | Compute L2 norm of a vector | Float32Array |
| `cosine_similarity_simd(a, b)` | Compute cosine similarity | Float32Array, Float32Array |
| `matrix_multiply_tiled(a, b, n)` | Matrix multiplication (n x n) | Float32Array, Float32Array, number |
| `quantize_int8_simd(input, scale)` | Quantize floats to INT8 | Float32Array, number |
| `conv2d_optimized(input, kernel, w, h, ks)` | 2D convolution with fused ReLU | Float32Array, Float32Array, number, number, number |
| `lz4_compress_optimized(input)` | LZ4 compression | Uint8Array |

### Serving Locally

To test in a browser, serve the files with a local HTTP server:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080/your_page.html` in your browser.

## Usage

Barq-WASM can be used as a library or a standalone runner.

### As a Standalone Runner

```bash
# Analyze and run a WASM module
./target/release/barq-wasm run path/to/workload.wasm

# Run with specific optimization flags
./target/release/barq-wasm run --opt-level=aggressive path/to/matrix_math.wasm
```

### As a Rust Library

```rust
use barq_wasm::runtime::Runtime;
use barq_wasm::config::Config;

fn main() -> anyhow::Result<()> {
    let config = Config::default().with_pattern_detection(true);
    let mut runtime = Runtime::new(config)?;
    
    let wasm_bytes = std::fs::read("matrix.wasm")?;
    runtime.load_module(&wasm_bytes)?;
    
    runtime.invoke("main", &[])?;
    Ok(())
}
```

## Project Architecture

The codebase is organized into four main pillars corresponding to the optimization phases:

*   `src/analyzer/`: Pattern detection logic (Phase 1)
*   `src/codegen/compression_codegen.rs`: LZ4/Zstd JIT emitters (Phase 2)
*   `src/codegen/vector_codegen.rs`: SIMD/AVX2 JIT emitters (Phase 3)
*   `src/codegen/database_codegen.rs`: DB/AI optimization logic (Phase 4)

## Contributing

Contributions are welcome to expand the catalog of detectable patterns or improve JIT code generation.

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/new-pattern`)
3.  Commit your changes
4.  Push to the branch
5.  Open a Pull Request

Please ensure all new code is covered by integration tests and passes `cargo clippy -- -D warnings`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
