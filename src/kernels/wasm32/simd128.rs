//! Explicit WebAssembly SIMD128 kernels.
//!
//! Every function here executes real `v128` instructions
//! (`f32x4.mul`, `f32x4.add`, `i32x4.trunc_sat_f32x4_s`, ...). Scalar tails
//! handle remainder elements. `scripts/verify-wasm-simd.sh` disassembles the
//! built bundle and fails if these instructions are missing.

use core::arch::wasm32::*;

/// Dot product over 4-lane f32 vectors with two accumulators.
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;
    unsafe {
        while i + 8 <= len {
            let va0 = v128_load(a.as_ptr().add(i) as *const v128);
            let vb0 = v128_load(b.as_ptr().add(i) as *const v128);
            let va1 = v128_load(a.as_ptr().add(i + 4) as *const v128);
            let vb1 = v128_load(b.as_ptr().add(i + 4) as *const v128);
            acc0 = f32x4_add(acc0, f32x4_mul(va0, vb0));
            acc1 = f32x4_add(acc1, f32x4_mul(va1, vb1));
            i += 8;
        }
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            acc0 = f32x4_add(acc0, f32x4_mul(va, vb));
            i += 4;
        }
    }
    let acc = f32x4_add(acc0, acc1);
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Sum of squares (L2 norm before the square root).
pub fn sum_of_squares_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let mut acc = f32x4_splat(0.0);
    let mut i = 0;
    unsafe {
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(va, va));
            i += 4;
        }
    }
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    while i < len {
        sum += a[i] * a[i];
        i += 1;
    }
    sum
}

/// L2 norm.
pub fn l2_norm_f32(a: &[f32]) -> f32 {
    sum_of_squares_f32(a).sqrt()
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

/// Quantize f32 to i8 with the crate-wide policy: IEEE round-to-nearest-even
/// of `x / scale`, saturating to [-128, 127], NaN -> 0.
///
/// `f32x4.nearest` implements ties-to-even and `i32x4.trunc_sat_f32x4_s`
/// saturates and maps NaN to 0, so the SIMD path matches the scalar
/// reference bit-for-bit.
pub fn quantize_f32_to_i8(input: &[f32], scale: f32, output: &mut Vec<i8>) {
    output.clear();
    output.reserve(input.len());
    let len = input.len();
    let vscale = f32x4_splat(scale);
    let vmax = i32x4_splat(127);
    let vmin = i32x4_splat(-128);
    let mut i = 0;
    let mut bytes = [0i8; 16];
    unsafe {
        while i + 8 <= len {
            let x0 = v128_load(input.as_ptr().add(i) as *const v128);
            let x1 = v128_load(input.as_ptr().add(i + 4) as *const v128);
            let r0 = f32x4_nearest(f32x4_div(x0, vscale));
            let r1 = f32x4_nearest(f32x4_div(x1, vscale));
            let q0 = i32x4_max(i32x4_min(i32x4_trunc_sat_f32x4(r0), vmax), vmin);
            let q1 = i32x4_max(i32x4_min(i32x4_trunc_sat_f32x4(r1), vmax), vmin);
            // Saturating narrows can't change values already in [-128, 127].
            let sixteen = i16x8_narrow_i32x4(q0, q1);
            let eight = i8x16_narrow_i16x8(sixteen, sixteen);
            v128_store(bytes.as_mut_ptr() as *mut v128, eight);
            output.extend_from_slice(&bytes[..8]);
            i += 8;
        }
    }
    while i < len {
        let scaled = input[i] / scale;
        output.push(scaled.round_ties_even().clamp(-128.0, 127.0) as i8);
        i += 1;
    }
}

/// Dequantize i8 to f32: `q as f32 * scale`, 16 lanes per iteration.
pub fn dequantize_i8_to_f32(input: &[i8], scale: f32, output: &mut Vec<f32>) {
    output.clear();
    output.resize(input.len(), 0.0);
    let len = input.len();
    let vscale = f32x4_splat(scale);
    let mut i = 0;
    unsafe {
        while i + 16 <= len {
            let bytes = v128_load(input.as_ptr().add(i) as *const v128);
            let lo16 = i16x8_extend_low_i8x16(bytes);
            let hi16 = i16x8_extend_high_i8x16(bytes);
            for (off, half) in [(0usize, lo16), (8usize, hi16)] {
                let lo32 = i32x4_extend_low_i16x8(half);
                let hi32 = i32x4_extend_high_i16x8(half);
                let flo = f32x4_mul(f32x4_convert_i32x4(lo32), vscale);
                let fhi = f32x4_mul(f32x4_convert_i32x4(hi32), vscale);
                v128_store(output.as_mut_ptr().add(i + off) as *mut v128, flo);
                v128_store(output.as_mut_ptr().add(i + off + 4) as *mut v128, fhi);
            }
            i += 16;
        }
    }
    while i < len {
        output[i] = f32::from(input[i]) * scale;
        i += 1;
    }
}

/// Elementwise addition.
pub fn vector_add_f32(a: &[f32], b: &[f32], output: &mut Vec<f32>) {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    output.clear();
    output.resize(len, 0.0);
    let mut i = 0;
    unsafe {
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            v128_store(output.as_mut_ptr().add(i) as *mut v128, f32x4_add(va, vb));
            i += 4;
        }
    }
    while i < len {
        output[i] = a[i] + b[i];
        i += 1;
    }
}

/// Elementwise multiplication.
pub fn vector_mul_f32(a: &[f32], b: &[f32], output: &mut Vec<f32>) {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    output.clear();
    output.resize(len, 0.0);
    let mut i = 0;
    unsafe {
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            v128_store(output.as_mut_ptr().add(i) as *mut v128, f32x4_mul(va, vb));
            i += 4;
        }
    }
    while i < len {
        output[i] = a[i] * b[i];
        i += 1;
    }
}
