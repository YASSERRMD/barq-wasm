//! Scalar reference kernels.
//!
//! These are the ground truth every SIMD implementation is tested against.
//! They favor clarity over speed; no unrolling, no unsafe.

use crate::error::{BarqError, BarqResult};

pub(crate) fn check_equal_len(a: &[f32], b: &[f32]) -> BarqResult<()> {
    if a.len() != b.len() {
        return Err(BarqError::InvalidArgument(format!(
            "slice lengths differ: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(())
}

/// Dot product of two equal-length vectors.
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean (L2) norm.
pub fn l2_norm_f32(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity; 0.0 when either vector has zero norm.
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_f32(a, b);
    let na = l2_norm_f32(a);
    let nb = l2_norm_f32(b);
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// Quantize `x` to i8: IEEE round-to-nearest-even of `x / scale`, saturating
/// to [-128, 127]. NaN quantizes to 0 (Rust saturating float->int cast).
/// This exact policy is implemented by the AVX2 and NEON kernels too, so
/// differential tests require bit-identical output.
pub fn quantize_f32_to_i8(input: &[f32], scale: f32, output: &mut Vec<i8>) {
    output.clear();
    output.reserve(input.len());
    for &x in input {
        let scaled = x / scale;
        let rounded = scaled.round_ties_even();
        output.push(rounded.clamp(-128.0, 127.0) as i8);
    }
}

/// Dequantize i8 back to f32: `q as f32 * scale`.
pub fn dequantize_i8_to_f32(input: &[i8], scale: f32, output: &mut Vec<f32>) {
    output.clear();
    output.reserve(input.len());
    for &q in input {
        output.push(f32::from(q) * scale);
    }
}

/// Row-major matrix multiply: (m x k) * (k x n) -> (m x n).
pub fn matrix_multiply_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            for j in 0..n {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
    c
}

/// Valid (no-padding) 2D convolution of a `w x h` single-channel input with a
/// square `ksize x ksize` kernel. Output is `(w-ksize+1) x (h-ksize+1)`.
pub fn conv2d_f32(input: &[f32], w: usize, h: usize, kernel: &[f32], ksize: usize) -> Vec<f32> {
    let out_w = w + 1 - ksize;
    let out_h = h + 1 - ksize;
    let mut out = vec![0.0f32; out_w * out_h];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut acc = 0.0f32;
            for ky in 0..ksize {
                for kx in 0..ksize {
                    acc += input[(oy + ky) * w + (ox + kx)] * kernel[ky * ksize + kx];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }
    out
}
