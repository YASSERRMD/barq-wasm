//! Differential tests: every SIMD kernel vs the scalar reference.
//!
//! Covers boundary lengths (empty, one element, below/at/above SIMD width),
//! large arrays, signs, zeros, NaN/infinity policy, misalignment, mismatched
//! lengths, and extreme quantization values. SIMD backends are exercised on
//! whatever the host CPU supports; unsupported backends must return typed
//! errors, never wrong results.

#![cfg(not(target_arch = "wasm32"))]

use barq_wasm::error::BarqError;
use barq_wasm::kernels::*;

const LENGTHS: &[usize] = &[
    0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1024, 10_007,
];

fn pseudo_random_f32(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 33) as f64) / ((1u64 << 31) as f64);
            (lo as f64 + unit * (hi - lo) as f64) as f32
        })
        .collect()
}

fn assert_rel_eq(a: f32, b: f32, max_rel: f32, ctx: &str) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(diff <= max_rel * scale, "{ctx}: {a} vs {b} (diff {diff})");
}

type DotFn = fn(&[f32], &[f32]) -> barq_wasm::error::BarqResult<f32>;
type NamedDotImpl = (&'static str, DotFn);

/// All dot-product implementations available on this host, with names.
fn dot_impls() -> Vec<NamedDotImpl> {
    let caps = cpu_capabilities();
    let mut impls: Vec<NamedDotImpl> = vec![];
    if caps.avx2 {
        impls.push(("avx2", dot_product_avx2 as _));
    }
    if caps.avx2 && caps.fma {
        impls.push(("avx2_fma", dot_product_avx2_fma as _));
    }
    if caps.neon {
        impls.push(("neon", dot_product_neon as _));
    }
    impls
}

#[test]
fn at_least_one_simd_backend_runs_on_supported_hosts() {
    let caps = cpu_capabilities();
    if caps.avx2 || caps.neon {
        assert!(
            !dot_impls().is_empty(),
            "host reports SIMD support but no SIMD impl is testable"
        );
    }
    let exec = dot_product_ex(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    println!("Selected backend: {}", exec.backend);
    if caps.avx2 || caps.neon {
        assert_ne!(
            exec.backend,
            KernelBackend::Scalar,
            "dispatch must pick SIMD when the CPU supports it"
        );
    }
}

#[test]
fn dot_product_matches_scalar_across_lengths() {
    for &len in LENGTHS {
        let a = pseudo_random_f32(len, 11, -100.0, 100.0);
        let b = pseudo_random_f32(len, 12, -100.0, 100.0);
        let reference = dot_product_scalar(&a, &b).unwrap();
        for (name, f) in dot_impls() {
            let got = f(&a, &b).unwrap();
            assert_rel_eq(reference, got, 1e-4, &format!("dot {name} len={len}"));
        }
        let auto = dot_product(&a, &b).unwrap();
        assert_rel_eq(reference, auto, 1e-4, &format!("dot auto len={len}"));
    }
}

#[test]
fn dot_product_signs_and_zeros() {
    let zeros = vec![0.0f32; 33];
    let mixed: Vec<f32> = (0..33)
        .map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 })
        .collect();
    for (name, f) in dot_impls() {
        assert_eq!(f(&zeros, &mixed).unwrap(), 0.0, "{name} zeros");
        let r = f(&mixed, &mixed).unwrap();
        let s = dot_product_scalar(&mixed, &mixed).unwrap();
        assert_rel_eq(s, r, 1e-5, &format!("{name} mixed signs"));
    }
}

#[test]
fn dot_product_nan_propagates() {
    let mut a = vec![1.0f32; 16];
    a[5] = f32::NAN;
    let b = vec![2.0f32; 16];
    assert!(dot_product_scalar(&a, &b).unwrap().is_nan());
    for (name, f) in dot_impls() {
        assert!(f(&a, &b).unwrap().is_nan(), "{name} must propagate NaN");
    }
}

#[test]
fn dot_product_infinity_policy() {
    let mut a = vec![1.0f32; 16];
    a[0] = f32::INFINITY;
    let b = vec![2.0f32; 16];
    let reference = dot_product_scalar(&a, &b).unwrap();
    assert_eq!(reference, f32::INFINITY);
    for (name, f) in dot_impls() {
        assert_eq!(f(&a, &b).unwrap(), f32::INFINITY, "{name} infinity");
    }
}

#[test]
fn dot_product_misaligned_slices() {
    // Slices offset by 1..4 elements from an allocation are still correct
    // (kernels use unaligned loads).
    let base_a = pseudo_random_f32(1024 + 4, 21, -10.0, 10.0);
    let base_b = pseudo_random_f32(1024 + 4, 22, -10.0, 10.0);
    for offset in 0..4usize {
        let a = &base_a[offset..offset + 1024];
        let b = &base_b[offset..offset + 1024];
        let reference = dot_product_scalar(a, b).unwrap();
        for (name, f) in dot_impls() {
            let got = f(a, b).unwrap();
            assert_rel_eq(reference, got, 1e-4, &format!("dot {name} offset={offset}"));
        }
    }
}

#[test]
fn mismatched_lengths_are_typed_errors() {
    let a = vec![1.0f32; 8];
    let b = vec![1.0f32; 9];
    assert!(matches!(
        dot_product_scalar(&a, &b),
        Err(BarqError::InvalidArgument(_))
    ));
    assert!(matches!(
        dot_product(&a, &b),
        Err(BarqError::InvalidArgument(_))
    ));
    for (_, f) in dot_impls() {
        assert!(matches!(f(&a, &b), Err(BarqError::InvalidArgument(_))));
    }
}

#[test]
fn l2_norm_and_cosine_match_scalar() {
    let caps = cpu_capabilities();
    for &len in LENGTHS {
        let a = pseudo_random_f32(len, 31, -50.0, 50.0);
        let b = pseudo_random_f32(len, 32, -50.0, 50.0);
        let norm_ref = l2_norm_scalar(&a).unwrap();
        let cos_ref = cosine_similarity_scalar(&a, &b).unwrap();
        if caps.avx2 {
            assert_rel_eq(
                norm_ref,
                l2_norm_avx2(&a).unwrap(),
                1e-4,
                &format!("norm avx2 len={len}"),
            );
            assert_rel_eq(
                cos_ref,
                cosine_similarity_avx2(&a, &b).unwrap(),
                1e-3,
                &format!("cos avx2 len={len}"),
            );
        }
        if caps.neon {
            assert_rel_eq(
                norm_ref,
                l2_norm_neon(&a).unwrap(),
                1e-4,
                &format!("norm neon len={len}"),
            );
            assert_rel_eq(
                cos_ref,
                cosine_similarity_neon(&a, &b).unwrap(),
                1e-3,
                &format!("cos neon len={len}"),
            );
        }
        assert_rel_eq(
            norm_ref,
            l2_norm(&a).unwrap(),
            1e-4,
            &format!("norm auto len={len}"),
        );
        assert_rel_eq(
            cos_ref,
            cosine_similarity(&a, &b).unwrap(),
            1e-3,
            &format!("cos auto len={len}"),
        );
    }
}

#[test]
fn cosine_zero_vector_is_zero_everywhere() {
    let caps = cpu_capabilities();
    let zeros = vec![0.0f32; 20];
    let v = pseudo_random_f32(20, 41, -5.0, 5.0);
    assert_eq!(cosine_similarity_scalar(&zeros, &v).unwrap(), 0.0);
    assert_eq!(cosine_similarity(&zeros, &v).unwrap(), 0.0);
    if caps.avx2 {
        assert_eq!(cosine_similarity_avx2(&zeros, &v).unwrap(), 0.0);
    }
    if caps.neon {
        assert_eq!(cosine_similarity_neon(&zeros, &v).unwrap(), 0.0);
    }
}

#[test]
fn quantize_is_bit_exact_against_scalar() {
    let caps = cpu_capabilities();
    for &len in LENGTHS {
        // Include values near ties (x.5 after scaling) and out-of-range.
        let mut input = pseudo_random_f32(len, 51, -300.0, 300.0);
        for (i, v) in input.iter_mut().enumerate() {
            if i % 7 == 0 {
                *v = (i as f32 / 2.0) - 64.0; // exact .5 ties after /1.0
            }
        }
        for &scale in &[1.0f32, 0.5, 0.037, 2.5] {
            let reference = quantize_i8_scalar(&input, scale).unwrap();
            if caps.avx2 {
                assert_eq!(
                    reference,
                    quantize_i8_avx2(&input, scale).unwrap(),
                    "avx2 len={len} scale={scale}"
                );
            }
            if caps.neon {
                assert_eq!(
                    reference,
                    quantize_i8_neon(&input, scale).unwrap(),
                    "neon len={len} scale={scale}"
                );
            }
            assert_eq!(
                reference,
                quantize_i8(&input, scale).unwrap(),
                "auto len={len} scale={scale}"
            );
        }
    }
}

#[test]
fn quantize_extreme_and_special_values() {
    let caps = cpu_capabilities();
    let input = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e30,
        -1e30,
        127.4,
        127.5,
        128.0,
        -128.5,
        -129.0,
        0.0,
        -0.0,
        0.5,
        1.5,
        2.5,
        -0.5,
        -1.5,
        -2.5,
    ];
    let reference = quantize_i8_scalar(&input, 1.0).unwrap();
    // Spot-check the documented policy directly.
    assert_eq!(reference[0], 0, "NaN quantizes to 0");
    assert_eq!(reference[1], 127, "+inf saturates");
    assert_eq!(reference[2], -128, "-inf saturates");
    assert_eq!(reference[12], 0, "0.5 rounds to even (0)");
    assert_eq!(reference[13], 2, "1.5 rounds to even (2)");
    assert_eq!(reference[14], 2, "2.5 rounds to even (2)");
    if caps.avx2 {
        assert_eq!(reference, quantize_i8_avx2(&input, 1.0).unwrap());
    }
    if caps.neon {
        assert_eq!(reference, quantize_i8_neon(&input, 1.0).unwrap());
    }
}

#[test]
fn quantize_zero_scale_is_typed_error() {
    let input = vec![1.0f32; 4];
    for scale in [0.0f32, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            quantize_i8(&input, scale),
            Err(BarqError::InvalidArgument(_))
        ));
    }
}

#[test]
fn dequantize_is_bit_exact_against_scalar() {
    let caps = cpu_capabilities();
    for &len in LENGTHS {
        let input: Vec<i8> = (0..len)
            .map(|i| ((i * 37 + 11) % 256) as u8 as i8)
            .collect();
        for &scale in &[1.0f32, 0.125, 0.037] {
            let reference = dequantize_i8_scalar(&input, scale).unwrap();
            if caps.avx2 {
                assert_eq!(
                    reference,
                    dequantize_i8_avx2(&input, scale).unwrap(),
                    "avx2 len={len}"
                );
            }
            if caps.neon {
                assert_eq!(
                    reference,
                    dequantize_i8_neon(&input, scale).unwrap(),
                    "neon len={len}"
                );
            }
            assert_eq!(
                reference,
                dequantize_i8(&input, scale).unwrap(),
                "auto len={len}"
            );
        }
    }
}

#[test]
fn quantize_dequantize_roundtrip_within_half_step() {
    let input = pseudo_random_f32(1000, 61, -100.0, 100.0);
    let scale = 1.0f32;
    let q = quantize_i8(&input, scale).unwrap();
    let d = dequantize_i8(&q, scale).unwrap();
    for (x, y) in input.iter().zip(d.iter()) {
        if x.abs() <= 127.0 {
            assert!((x - y).abs() <= scale * 0.5 + 1e-6, "roundtrip {x} -> {y}");
        }
    }
}

#[test]
fn matrix_multiply_matches_scalar() {
    let caps = cpu_capabilities();
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (2, 3, 4),
        (4, 4, 4),
        (7, 5, 9),
        (8, 8, 8),
        (16, 16, 16),
        (31, 33, 17),
        (64, 64, 64),
    ] {
        let a = pseudo_random_f32(m * k, 71, -2.0, 2.0);
        let b = pseudo_random_f32(k * n, 72, -2.0, 2.0);
        let reference = matrix_multiply_scalar(&a, &b, m, k, n).unwrap();
        if caps.avx2 && caps.fma {
            let got = matrix_multiply_avx2_fma(&a, &b, m, k, n).unwrap();
            for i in 0..m * n {
                assert_rel_eq(
                    reference[i],
                    got[i],
                    1e-4,
                    &format!("matmul avx2fma {m}x{k}x{n} idx={i}"),
                );
            }
        }
        if caps.neon {
            let got = matrix_multiply_neon(&a, &b, m, k, n).unwrap();
            for i in 0..m * n {
                assert_rel_eq(
                    reference[i],
                    got[i],
                    1e-4,
                    &format!("matmul neon {m}x{k}x{n} idx={i}"),
                );
            }
        }
        let auto = matrix_multiply(&a, &b, m, k, n).unwrap();
        for i in 0..m * n {
            assert_rel_eq(
                reference[i],
                auto[i],
                1e-4,
                &format!("matmul auto {m}x{k}x{n} idx={i}"),
            );
        }
    }
}

#[test]
fn matrix_dims_mismatch_is_typed_error() {
    let a = vec![0.0f32; 6];
    let b = vec![0.0f32; 6];
    assert!(matches!(
        matrix_multiply(&a, &b, 2, 4, 2),
        Err(BarqError::InvalidArgument(_))
    ));
}

#[test]
fn conv2d_reference_known_answer() {
    // 3x3 input, 2x2 kernel of ones -> each output is the sum of a 2x2 patch.
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let kernel = vec![1.0f32; 4];
    let out = conv2d(&input, 3, 3, &kernel, 2).unwrap();
    assert_eq!(out, vec![12.0, 16.0, 24.0, 28.0]);
    let exec = conv2d_ex(&input, 3, 3, &kernel, 2).unwrap();
    assert_eq!(
        exec.backend,
        KernelBackend::Scalar,
        "conv2d is scalar-only and must say so"
    );
}
