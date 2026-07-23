/* tslint:disable */
/* eslint-disable */

/**
 * Argmax: index of maximum value with 4-wide tracking
 */
export function argmax(a: Float32Array): number;

/**
 * Argmin: index of minimum value with 4-wide tracking
 */
export function argmin(a: Float32Array): number;

/**
 * Average pooling 2D
 */
export function avg_pooling_2d(input: Float32Array, width: number, height: number, pool_size: number): Float32Array;

/**
 * Batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
 */
export function batch_normalize(input: Float32Array, gamma: number, beta: number, epsilon: number): Float32Array;

/**
 * Identity buffer copy. This performs NO compression; it exists only as a
 * memcpy-cost baseline for benchmarks.
 */
export function buffer_copy_baseline(input: Uint8Array): Uint8Array;

/**
 * Conv2D with a fully unrolled 3x3 fast path (scalar arithmetic).
 */
export function conv2d_optimized(input: Float32Array, kernel: Float32Array, width: number, height: number, kernel_size: number): Float32Array;

/**
 * Scalar convolution (baseline)
 */
export function conv2d_scalar(input: Float32Array, kernel: Float32Array, width: number, height: number, kernel_size: number): Float32Array;

/**
 * Scalar cosine similarity (baseline)
 */
export function cosine_similarity_scalar(a: Float32Array, b: Float32Array): number;

/**
 * Cosine similarity executing explicit `v128`/`f32x4` instructions.
 */
export function cosine_similarity_simd128(a: Float32Array, b: Float32Array): number;

/**
 * Cosine similarity built from the unrolled scalar dot product and norm.
 */
export function cosine_similarity_unrolled_scalar(a: Float32Array, b: Float32Array): number;

/**
 * Dequantize INT8 back to float32
 */
export function dequantize_int8(input: Int8Array, scale: number): Float32Array;

/**
 * INT8 dequantization executing `i16x8`/`i32x4` extends + `f32x4.mul`.
 */
export function dequantize_int8_simd128(input: Int8Array, scale: number): Float32Array;

/**
 * Scalar dot product (baseline)
 */
export function dot_product_scalar(a: Float32Array, b: Float32Array): number;

/**
 * Dot product executing explicit `v128`/`f32x4` instructions.
 */
export function dot_product_simd128(a: Float32Array, b: Float32Array): number;

/**
 * Dot product with 16-wide manual unrolling and unsafe pointer access.
 * Uses 16 independent scalar accumulators for instruction-level parallelism.
 * This is NOT a SIMD implementation.
 */
export function dot_product_unrolled_scalar(a: Float32Array, b: Float32Array): number;

/**
 * Euclidean distance between two vectors
 */
export function euclidean_distance(a: Float32Array, b: Float32Array): number;

/**
 * Leaky ReLU activation function with 8-wide unrolling
 */
export function leaky_relu(a: Float32Array, alpha: number): Float32Array;

/**
 * Experimental LZ4-style compressor.
 *
 * Honest description of what this actually does:
 * - Buffers < 128 KiB are returned **verbatim** — no compression happens and
 *   the output is NOT valid LZ4 data.
 * - Larger buffers go through a hash-based LZ4-like block emitter that has
 *   no matching decompressor in this crate and has never been validated
 *   against the LZ4 format specification.
 *
 * Do not use this where real LZ4 output is required.
 */
export function lz4_compress_experimental(input: Uint8Array): Uint8Array;

/**
 * Manhattan distance (L1 norm) between two vectors
 */
export function manhattan_distance(a: Float32Array, b: Float32Array): number;

/**
 * Matrix addition: C = A + B
 */
export function matrix_add(a: Float32Array, b: Float32Array): Float32Array;

/**
 * Scalar matrix multiplication (baseline)
 */
export function matrix_multiply_scalar(a: Float32Array, b: Float32Array, n: number): Float32Array;

/**
 * Matrix multiplication with multi-level cache tiling (scalar arithmetic).
 * Uses 32x32 tiles, processed in k-i-j order for row-major access.
 */
export function matrix_multiply_tiled(a: Float32Array, b: Float32Array, n: number): Float32Array;

/**
 * Matrix scalar multiplication: C = A * scalar
 */
export function matrix_scalar_multiply(a: Float32Array, scalar: number): Float32Array;

/**
 * Matrix transpose (n x m -> m x n) with unsafe pointer access for speed
 */
export function matrix_transpose(a: Float32Array, rows: number, cols: number): Float32Array;

/**
 * Max pooling 2D (stride = kernel_size for non-overlapping)
 */
export function max_pooling_2d(input: Float32Array, width: number, height: number, pool_size: number): Float32Array;

/**
 * Compute mean of a vector
 */
export function mean(a: Float32Array): number;

/**
 * Scalar INT8 quantization (baseline)
 */
export function quantize_int8_scalar(input: Float32Array, scale: number): Int8Array;

/**
 * INT8 quantization executing `f32x4.nearest` + `i32x4.trunc_sat_f32x4_s`.
 * Policy: round-to-nearest-even of x/scale, saturating, NaN -> 0 —
 * bit-identical to `kernels::scalar::quantize_f32_to_i8`.
 */
export function quantize_int8_simd128(input: Float32Array, scale: number): Int8Array;

/**
 * INT8 quantization, manually unrolled to process 16 elements per iteration.
 * This is a scalar implementation: it does NOT use `v128`/`f32x4`
 * instructions or any other explicit SIMD.
 */
export function quantize_int8_unrolled_scalar(input: Float32Array, scale: number): Int8Array;

/**
 * ReLU activation function
 */
export function relu(a: Float32Array): Float32Array;

/**
 * Sigmoid activation function
 */
export function sigmoid(a: Float32Array): Float32Array;

/**
 * True when this bundle was compiled with the wasm `simd128` target feature.
 * JavaScript can use this to prove which bundle it loaded.
 */
export function simd128_enabled(): boolean;

/**
 * Softmax function (numerically stable)
 */
export function softmax(a: Float32Array): Float32Array;

/**
 * Compute standard deviation
 */
export function std_dev(a: Float32Array): number;

/**
 * Compute variance of a vector
 */
export function variance(a: Float32Array): number;

/**
 * Vector addition: c = a + b
 */
export function vector_add(a: Float32Array, b: Float32Array): Float32Array;

/**
 * Elementwise addition executing `f32x4.add`.
 */
export function vector_add_simd128(a: Float32Array, b: Float32Array): Float32Array;

/**
 * Clamp vector values between min and max
 */
export function vector_clamp(a: Float32Array, min_val: number, max_val: number): Float32Array;

/**
 * Element-wise multiplication (Hadamard product): c = a * b
 */
export function vector_elementwise_multiply(a: Float32Array, b: Float32Array): Float32Array;

/**
 * Elementwise multiplication executing `f32x4.mul`.
 */
export function vector_elementwise_multiply_simd128(a: Float32Array, b: Float32Array): Float32Array;

/**
 * Find maximum value in a vector with 8-wide unrolling
 */
export function vector_max(a: Float32Array): number;

/**
 * Find minimum value in a vector with 8-wide unrolling
 */
export function vector_min(a: Float32Array): number;

/**
 * Scalar vector norm (baseline)
 */
export function vector_norm_scalar(a: Float32Array): number;

/**
 * L2 norm executing explicit `v128`/`f32x4` instructions.
 */
export function vector_norm_simd128(a: Float32Array): number;

/**
 * L2 norm with 8-wide manual unrolling (scalar arithmetic, not SIMD).
 */
export function vector_norm_unrolled_scalar(a: Float32Array): number;

/**
 * Normalize vector to unit length
 */
export function vector_normalize(a: Float32Array): Float32Array;

/**
 * Vector scaling: c = a * scalar
 */
export function vector_scale(a: Float32Array, scalar: number): Float32Array;

/**
 * Vector subtraction: c = a - b
 */
export function vector_subtract(a: Float32Array, b: Float32Array): Float32Array;

/**
 * Sum of all elements in a vector
 */
export function vector_sum(a: Float32Array): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly argmax: (a: number, b: number) => number;
    readonly argmin: (a: number, b: number) => number;
    readonly avg_pooling_2d: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly batch_normalize: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly buffer_copy_baseline: (a: number, b: number) => [number, number];
    readonly conv2d_optimized: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly conv2d_scalar: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly cosine_similarity_scalar: (a: number, b: number, c: number, d: number) => number;
    readonly cosine_similarity_unrolled_scalar: (a: number, b: number, c: number, d: number) => number;
    readonly dequantize_int8: (a: number, b: number, c: number) => [number, number];
    readonly dot_product_scalar: (a: number, b: number, c: number, d: number) => number;
    readonly dot_product_unrolled_scalar: (a: number, b: number, c: number, d: number) => number;
    readonly euclidean_distance: (a: number, b: number, c: number, d: number) => number;
    readonly leaky_relu: (a: number, b: number, c: number) => [number, number];
    readonly lz4_compress_experimental: (a: number, b: number) => [number, number];
    readonly manhattan_distance: (a: number, b: number, c: number, d: number) => number;
    readonly matrix_add: (a: number, b: number, c: number, d: number) => [number, number];
    readonly matrix_multiply_scalar: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly matrix_multiply_tiled: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly matrix_scalar_multiply: (a: number, b: number, c: number) => [number, number];
    readonly matrix_transpose: (a: number, b: number, c: number, d: number) => [number, number];
    readonly max_pooling_2d: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly mean: (a: number, b: number) => number;
    readonly quantize_int8_scalar: (a: number, b: number, c: number) => [number, number];
    readonly quantize_int8_unrolled_scalar: (a: number, b: number, c: number) => [number, number];
    readonly relu: (a: number, b: number) => [number, number];
    readonly sigmoid: (a: number, b: number) => [number, number];
    readonly simd128_enabled: () => number;
    readonly softmax: (a: number, b: number) => [number, number];
    readonly std_dev: (a: number, b: number) => number;
    readonly variance: (a: number, b: number) => number;
    readonly vector_add: (a: number, b: number, c: number, d: number) => [number, number];
    readonly vector_clamp: (a: number, b: number, c: number, d: number) => [number, number];
    readonly vector_elementwise_multiply: (a: number, b: number, c: number, d: number) => [number, number];
    readonly vector_max: (a: number, b: number) => number;
    readonly vector_min: (a: number, b: number) => number;
    readonly vector_norm_scalar: (a: number, b: number) => number;
    readonly vector_norm_unrolled_scalar: (a: number, b: number) => number;
    readonly vector_normalize: (a: number, b: number) => [number, number];
    readonly vector_scale: (a: number, b: number, c: number) => [number, number];
    readonly vector_subtract: (a: number, b: number, c: number, d: number) => [number, number];
    readonly vector_sum: (a: number, b: number) => number;
    readonly cosine_similarity_simd128: (a: number, b: number, c: number, d: number) => number;
    readonly dequantize_int8_simd128: (a: number, b: number, c: number) => [number, number];
    readonly dot_product_simd128: (a: number, b: number, c: number, d: number) => number;
    readonly quantize_int8_simd128: (a: number, b: number, c: number) => [number, number];
    readonly vector_add_simd128: (a: number, b: number, c: number, d: number) => [number, number];
    readonly vector_elementwise_multiply_simd128: (a: number, b: number, c: number, d: number) => [number, number];
    readonly vector_norm_simd128: (a: number, b: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
