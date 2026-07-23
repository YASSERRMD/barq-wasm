//! NEON kernels for ARM64 (aarch64).
//!
//! Every function here executes explicit 128-bit NEON intrinsics. Callers
//! must verify NEON support before calling; the safe wrappers in
//! `kernels::mod` do exactly that.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::*;

/// Dot product with fused multiply-add over 4-lane f32 vectors.
///
/// # Safety
/// Requires NEON. `a` and `b` must have equal lengths.
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 8 <= len {
        let va0 = vld1q_f32(a.as_ptr().add(i));
        let vb0 = vld1q_f32(b.as_ptr().add(i));
        let va1 = vld1q_f32(a.as_ptr().add(i + 4));
        let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc0 = vfmaq_f32(acc0, va0, vb0);
        acc1 = vfmaq_f32(acc1, va1, vb1);
        i += 8;
    }
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, va, vb);
        i += 4;
    }
    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Sum of squares (the L2 norm before the square root).
///
/// # Safety
/// Requires NEON.
#[target_feature(enable = "neon")]
pub unsafe fn sum_of_squares_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        acc = vfmaq_f32(acc, va, va);
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
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
/// Requires NEON. `scale` must be non-zero (checked by the safe wrapper).
#[target_feature(enable = "neon")]
pub unsafe fn quantize_f32_to_i8(input: &[f32], scale: f32, output: &mut Vec<i8>) {
    output.clear();
    output.reserve(input.len());
    let len = input.len();
    let vscale = vdupq_n_f32(scale);
    let vmax = vdupq_n_s32(127);
    let vmin = vdupq_n_s32(-128);
    let mut i = 0;
    let mut lanes = [0i32; 4];
    while i + 4 <= len {
        let x = vld1q_f32(input.as_ptr().add(i));
        let scaled = vdivq_f32(x, vscale);
        // vcvtnq: round to nearest, ties to even. NaN converts to 0, and the
        // conversion saturates +/-inf, so no pre-clamp is needed.
        let ints = vcvtnq_s32_f32(scaled);
        let clamped = vmaxq_s32(vminq_s32(ints, vmax), vmin);
        vst1q_s32(lanes.as_mut_ptr(), clamped);
        for lane in lanes {
            output.push(lane as i8);
        }
        i += 4;
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
/// Requires NEON.
#[target_feature(enable = "neon")]
pub unsafe fn dequantize_i8_to_f32(input: &[i8], scale: f32, output: &mut Vec<f32>) {
    output.clear();
    output.resize(input.len(), 0.0);
    let len = input.len();
    let vscale = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 8 <= len {
        let bytes = vld1_s8(input.as_ptr().add(i));
        let wide = vmovl_s8(bytes); // 8 x i16
        let lo = vmovl_s16(vget_low_s16(wide)); // 4 x i32
        let hi = vmovl_s16(vget_high_s16(wide));
        let flo = vmulq_f32(vcvtq_f32_s32(lo), vscale);
        let fhi = vmulq_f32(vcvtq_f32_s32(hi), vscale);
        vst1q_f32(output.as_mut_ptr().add(i), flo);
        vst1q_f32(output.as_mut_ptr().add(i + 4), fhi);
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
/// Requires NEON. Slice lengths must match the dimensions (checked by the
/// safe wrapper).
#[target_feature(enable = "neon")]
pub unsafe fn matrix_multiply_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_ik = vdupq_n_f32(a[i * k + kk]);
            let b_row = b.as_ptr().add(kk * n);
            let c_row = c.as_mut_ptr().add(i * n);
            let mut j = 0;
            while j + 4 <= n {
                let vb = vld1q_f32(b_row.add(j));
                let vc = vld1q_f32(c_row.add(j));
                vst1q_f32(c_row.add(j), vfmaq_f32(vc, a_ik, vb));
                j += 4;
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
