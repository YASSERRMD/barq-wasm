//! AVX2 (and AVX2+FMA) kernels for x86-64.
//!
//! Every function here executes explicit 256-bit SIMD intrinsics. Callers
//! must verify AVX2 (and FMA where applicable) support before calling; the
//! safe wrappers in `kernels::mod` do exactly that.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

/// Horizontal sum of one 256-bit register of 8 f32 lanes.
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let hi2 = _mm_movehl_ps(shuf, sum2);
    let sum1 = _mm_add_ss(sum2, hi2);
    _mm_cvtss_f32(sum1)
}

/// Dot product using AVX2 multiply + add (no FMA).
///
/// # Safety
/// Requires AVX2. `a` and `b` must have equal lengths.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= len {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
        i += 16;
    }
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va, vb));
        i += 8;
    }
    let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Dot product using AVX2 with fused multiply-add.
///
/// # Safety
/// Requires AVX2 and FMA. `a` and `b` must have equal lengths.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_f32_fma(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= len {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        i += 16;
    }
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }
    let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Sum of squares (the L2 norm before the square root).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn sum_of_squares_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, va));
        i += 8;
    }
    let mut sum = hsum256(acc);
    while i < len {
        sum += a[i] * a[i];
        i += 1;
    }
    sum
}

/// Quantize f32 to i8: round-to-nearest-even of `x / scale`, saturating.
/// NaN inputs quantize to 0, matching the scalar reference.
///
/// # Safety
/// Requires AVX2. `scale` must be non-zero (checked by the safe wrapper).
#[target_feature(enable = "avx2")]
pub unsafe fn quantize_f32_to_i8(input: &[f32], scale: f32, output: &mut Vec<i8>) {
    output.clear();
    output.reserve(input.len());
    let len = input.len();
    let vscale = _mm256_set1_ps(scale);
    let vmax = _mm256_set1_epi32(127);
    let vmin = _mm256_set1_epi32(-128);
    let mut i = 0;
    let mut lanes = [0i32; 8];
    while i + 8 <= len {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let scaled = _mm256_div_ps(x, vscale);
        // Map +/-inf into the clampable range before conversion; NaN lanes
        // are zeroed below via the ordered-compare mask.
        let bounded = _mm256_min_ps(
            _mm256_max_ps(scaled, _mm256_set1_ps(-1.0e4)),
            _mm256_set1_ps(1.0e4),
        );
        let rounded = _mm256_round_ps(bounded, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let ints = _mm256_cvttps_epi32(rounded);
        let clamped = _mm256_max_epi32(_mm256_min_epi32(ints, vmax), vmin);
        let ordered = _mm256_castps_si256(_mm256_cmp_ps(scaled, scaled, _CMP_ORD_Q));
        let final_i32 = _mm256_and_si256(clamped, ordered);
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, final_i32);
        for lane in lanes {
            output.push(lane as i8);
        }
        i += 8;
    }
    while i < len {
        let scaled = input[i] / scale;
        let q = scaled.round_ties_even().clamp(-128.0, 127.0) as i8;
        output.push(q);
        i += 1;
    }
}

/// Dequantize i8 to f32: `q as f32 * scale`.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn dequantize_i8_to_f32(input: &[i8], scale: f32, output: &mut Vec<f32>) {
    output.clear();
    output.resize(input.len(), 0.0);
    let len = input.len();
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= len {
        // Sign-extend 8 i8 lanes to i32, convert to f32, scale.
        let bytes = _mm_loadl_epi64(input.as_ptr().add(i) as *const __m128i);
        let ints = _mm256_cvtepi8_epi32(bytes);
        let floats = _mm256_cvtepi32_ps(ints);
        let scaled = _mm256_mul_ps(floats, vscale);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), scaled);
        i += 8;
    }
    while i < len {
        output[i] = f32::from(input[i]) * scale;
        i += 1;
    }
}

/// Row-major matrix multiply (m x k) * (k x n) with an FMA-vectorized inner
/// loop over `j`.
///
/// # Safety
/// Requires AVX2 and FMA. Slice lengths must match the dimensions
/// (checked by the safe wrapper).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matrix_multiply_f32_fma(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_ik = _mm256_set1_ps(a[i * k + kk]);
            let b_row = b.as_ptr().add(kk * n);
            let c_row = c.as_mut_ptr().add(i * n);
            let mut j = 0;
            while j + 8 <= n {
                let vb = _mm256_loadu_ps(b_row.add(j));
                let vc = _mm256_loadu_ps(c_row.add(j));
                _mm256_storeu_ps(c_row.add(j), _mm256_fmadd_ps(a_ik, vb, vc));
                j += 8;
            }
            let a_s = a[i * k + kk];
            while j < n {
                *c_row.add(j) += a_s * *b_row.add(j);
                j += 1;
            }
        }
    }
    c
}
