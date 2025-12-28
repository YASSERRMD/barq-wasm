# WebAssembly (WASM) Integration & Usage

Barq-WASM provides a high-performance WebAssembly package optimized for browser-based numerical computation, AI/ML inference, and data processing.

## Prerequisites

- **Rust**: Latest stable toolchain.
- **wasm-pack**: The standard tool for building Rust-WASM packages.
  ```bash
  cargo install wasm-pack
  ```

## Building the Package

To build the WASM package for the web:

```bash
wasm-pack build --target web --features wasm
```

This will create a `pkg/` directory in your project root containing the WASM binary and JavaScript glue code.

## Quick Start (HTML/JavaScript)

Include the generated package in your HTML file as a module:

```html
<!DOCTYPE html>
<html>
<body>
    <script type="module">
        import init, { dot_product_simd, matrix_multiply_tiled } from './pkg/barq_wasm.js';

        async function run() {
            // Initialize the WASM module
            await init();

            // Vector Operations
            const a = new Float32Array([1.0, 2.0, 3.0, 4.0]);
            const b = new Float32Array([5.0, 6.0, 7.0, 8.0]);
            const dot = dot_product_simd(a, b);
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
| `vector_norm_simd(a)` | Compute L2 norm (SIMD) | `(Float32Array)` |
| `dot_product_simd(a, b)` | High-performance dot product | `(Float32Array, Float32Array)` |
| `cosine_similarity_simd(a, b)` | Cosine similarity between vectors | `(Float32Array, Float32Array)` |

### Matrix Operations
| Function | Description | Parameters |
|:---|:---|:---|
| `matrix_multiply_tiled(a, b, n)` | Cache-optimized matmul (n x n) | `(Float32Array, Float32Array, number)` |
| `matrix_transpose(a, r, c)` | Optimized matrix transpose | `(Float32Array, number, number)` |
| `matrix_add(a, b)` | Matrix addition | `(Float32Array, Float32Array)` |
| `matrix_scalar_multiply(a, s)` | Matrix-scalar multiplication | `(Float32Array, number)` |

### AI/ML Operations
| Function | Description | Parameters |
|:---|:---|:---|
| `quantize_int8_simd(input, scale)` | Fast INT8 quantization | `(Float32Array, number)` |
| `dequantize_int8(input, scale)` | Convert INT8 back to float | `(Int8Array, number)` |
| `conv2d_optimized(inp, k, w, h, ks)`| 2D convolution + Fused ReLU | `(Float32Array, Float32Array, ...)` |
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
| `lz4_compress_optimized(input)`| Adaptive LZ4 compression | `(Uint8Array)` |

## Benchmarking

A complete benchmark suite is available in the `examples/wasm` directory. To run it locally:

1. Build the WASM package: `wasm-pack build --target web --features wasm`
2. Start a local server: `python3 -m http.server 8080`
3. Open `http://localhost:8080/examples/wasm/benchmark.html`

## GitHub Pages

The benchmarks are automatically published to GitHub Pages upon every push to the `main` branch. You can view the live performance metrics at:
`https://<username>.github.io/barq-wasm/benchmark.html`
