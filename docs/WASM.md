# WebAssembly (WASM) Integration & Usage

Barq-WASM provides a WebAssembly package of **scalar** compute kernels for
browser-based numerical computation. No function in the current package
executes SIMD instructions; the `*_unrolled_scalar` variants are manually
unrolled scalar loops. Explicit WASM SIMD128 kernels are planned (Phase 4)
and will ship as a separate, verified bundle.

## Prerequisites

- **Rust**: Latest stable toolchain.
- **wasm-pack**: The standard tool for building Rust-WASM packages.
  ```bash
  cargo install wasm-pack
  ```

## Building the Package

To build the WASM package for the web:

```bash
wasm-pack build --target web --features wasm --no-default-features
```

This will create a `pkg/` directory in your project root containing the WASM binary and JavaScript glue code.

## Quick Start (HTML/JavaScript)

Include the generated package in your HTML file as a module:

```html
<!DOCTYPE html>
<html>
<body>
    <script type="module">
        import init, { dot_product_unrolled_scalar, matrix_multiply_tiled } from './pkg/barq_wasm.js';

        async function run() {
            // Initialize the WASM module
            await init();

            // Vector Operations
            const a = new Float32Array([1.0, 2.0, 3.0, 4.0]);
            const b = new Float32Array([5.0, 6.0, 7.0, 8.0]);
            const dot = dot_product_unrolled_scalar(a, b);
            console.log('Dot Product:', dot);

            // Matrix Operations
            const size = 64;
            const matA = new Float32Array(size * size).fill(0.5);
            const matB = new Float32Array(size * size).fill(1.0);
            const matC = matrix_multiply_tiled(matA, matB, size);
            console.log('Matrix result[0]:', matC[0]);
        }

        run();
    </script>
</body>
</html>
```

## API Reference

All functions are scalar implementations.

### Vector Operations
| Function | Description | Parameters |
|:---|:---|:---|
| `vector_add(a, b)` | Element-wise addition | `(Float32Array, Float32Array)` |
| `vector_subtract(a, b)` | Element-wise subtraction | `(Float32Array, Float32Array)` |
| `vector_scale(a, scalar)` | Multiply vector by a scalar | `(Float32Array, number)` |
| `vector_elementwise_multiply(a, b)` | Element-wise multiplication | `(Float32Array, Float32Array)` |
| `vector_sum(a)` | Sum of all elements | `(Float32Array)` |
| `vector_min(a)` | Minimum value in vector | `(Float32Array)` |
| `vector_max(a)` | Maximum value in vector | `(Float32Array)` |
| `vector_normalize(a)` | L2 normalization | `(Float32Array)` |
| `vector_norm_scalar(a)` | L2 norm, naive scalar | `(Float32Array)` |
| `vector_norm_unrolled_scalar(a)` | L2 norm, 8-wide unrolled scalar | `(Float32Array)` |
| `dot_product_scalar(a, b)` | Dot product, naive scalar | `(Float32Array, Float32Array)` |
| `dot_product_unrolled_scalar(a, b)` | Dot product, 16-wide unrolled scalar | `(Float32Array, Float32Array)` |
| `cosine_similarity_scalar(a, b)` | Cosine similarity, naive scalar | `(Float32Array, Float32Array)` |
| `cosine_similarity_unrolled_scalar(a, b)` | Cosine similarity, unrolled scalar | `(Float32Array, Float32Array)` |

### Matrix Operations
| Function | Description | Parameters |
|:---|:---|:---|
| `matrix_multiply_scalar(a, b, n)` | Naive matmul (n x n) | `(Float32Array, Float32Array, number)` |
| `matrix_multiply_tiled(a, b, n)` | Cache-tiled matmul (n x n) | `(Float32Array, Float32Array, number)` |
| `matrix_transpose(a, r, c)` | Matrix transpose | `(Float32Array, number, number)` |
| `matrix_add(a, b)` | Matrix addition | `(Float32Array, Float32Array)` |
| `matrix_scalar_multiply(a, s)` | Matrix-scalar multiplication | `(Float32Array, number)` |

### AI/ML Operations
| Function | Description | Parameters |
|:---|:---|:---|
| `quantize_int8_scalar(input, scale)` | INT8 quantization, naive scalar | `(Float32Array, number)` |
| `quantize_int8_unrolled_scalar(input, scale)` | INT8 quantization, unrolled scalar | `(Float32Array, number)` |
| `dequantize_int8(input, scale)` | Convert INT8 back to float | `(Int8Array, number)` |
| `conv2d_optimized(inp, k, w, h, ks)`| 2D convolution + fused ReLU (scalar, unrolled 3x3 path) | `(Float32Array, Float32Array, ...)` |
| `conv2d_scalar(inp, k, w, h, ks)` | 2D convolution, naive scalar | `(Float32Array, Float32Array, ...)` |
| `max_pooling_2d(input, w, h, p)` | 2D Max Pooling | `(Float32Array, number, number, number)` |
| `avg_pooling_2d(input, w, h, p)` | 2D Average Pooling | `(Float32Array, number, number, number)` |
| `batch_normalize(a, g, b, e)` | Batch Normalization | `(Float32Array, ...)` |
| `softmax(a)` | Softmax activation | `(Float32Array)` |
| `sigmoid(a)` | Sigmoid activation | `(Float32Array)` |
| `relu(a)` | ReLU activation | `(Float32Array)` |
| `leaky_relu(a, alpha)` | Leaky ReLU activation | `(Float32Array, number)` |
| `argmax(a)` | Index of maximum element | `(Float32Array)` |
| `argmin(a)` | Index of minimum element | `(Float32Array)` |

### Utility & Distance
| Function | Description | Parameters |
|:---|:---|:---|
| `euclidean_distance(a, b)` | L2 distance | `(Float32Array, Float32Array)` |
| `manhattan_distance(a, b)` | L1 distance | `(Float32Array, Float32Array)` |
| `vector_clamp(a, min, max)` | Clamp elements to range | `(Float32Array, number, number)` |
| `mean(a)` | Arithmetic mean | `(Float32Array)` |
| `variance(a)` | Sample variance | `(Float32Array)` |
| `std_dev(a)` | Standard deviation | `(Float32Array)` |
| `lz4_compress_experimental(input)`| Experimental LZ4-style compressor. Returns the input **verbatim** below 128 KiB; larger inputs use an unvalidated LZ4-like block format with no decompressor. | `(Uint8Array)` |
| `buffer_copy_baseline(input)` | Identity copy (benchmark baseline, no compression) | `(Uint8Array)` |

## Benchmarking

The previous browser benchmark pages were removed because their comparisons
were not reproducible or methodologically fair. A real browser benchmark
harness (warm-up, multiple samples, median/p95, environment metadata,
correctness checks) is Phase 7 scope.
