//! Property-based tests: randomized lengths, values, and scales, comparing
//! every available SIMD backend against the scalar reference.

#![cfg(not(target_arch = "wasm32"))]

use barq_wasm::kernels::*;
use proptest::prelude::*;

fn finite_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        -1000.0f32..1000.0f32,
        -1.0f32..1.0f32,
        Just(0.0f32),
        Just(-0.0f32),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn dot_product_all_backends_agree(
        pair in proptest::collection::vec((finite_f32(), finite_f32()), 0..600)
    ) {
        let a: Vec<f32> = pair.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = pair.iter().map(|(_, y)| *y).collect();
        let reference = dot_product_scalar(&a, &b).unwrap();
        let caps = cpu_capabilities();
        // Different valid summation orders diverge in proportion to the sum
        // of |a_i * b_i| (worst under cancellation), not to the result.
        let abs_sum: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
        let tol = 1e-4 * abs_sum + 1e-3;
        if caps.avx2 {
            prop_assert!((dot_product_avx2(&a, &b).unwrap() - reference).abs() <= tol);
        }
        if caps.avx2 && caps.fma {
            prop_assert!((dot_product_avx2_fma(&a, &b).unwrap() - reference).abs() <= tol);
        }
        if caps.neon {
            prop_assert!((dot_product_neon(&a, &b).unwrap() - reference).abs() <= tol);
        }
        prop_assert!((dot_product(&a, &b).unwrap() - reference).abs() <= tol);
    }

    #[test]
    fn l2_norm_all_backends_agree(a in proptest::collection::vec(finite_f32(), 0..600)) {
        let reference = l2_norm_scalar(&a).unwrap();
        let caps = cpu_capabilities();
        let tol = 1e-4 * reference.abs().max(1.0);
        if caps.avx2 {
            prop_assert!((l2_norm_avx2(&a).unwrap() - reference).abs() <= tol);
        }
        if caps.neon {
            prop_assert!((l2_norm_neon(&a).unwrap() - reference).abs() <= tol);
        }
    }

    #[test]
    fn quantize_all_backends_bit_exact(
        input in proptest::collection::vec(finite_f32(), 0..600),
        scale in prop_oneof![0.01f32..10.0f32, Just(1.0f32)],
    ) {
        let reference = quantize_i8_scalar(&input, scale).unwrap();
        let caps = cpu_capabilities();
        if caps.avx2 {
            prop_assert_eq!(&reference, &quantize_i8_avx2(&input, scale).unwrap());
        }
        if caps.neon {
            prop_assert_eq!(&reference, &quantize_i8_neon(&input, scale).unwrap());
        }
        prop_assert_eq!(&reference, &quantize_i8(&input, scale).unwrap());
    }

    #[test]
    fn dequantize_all_backends_bit_exact(
        input in proptest::collection::vec(any::<i8>(), 0..600),
        scale in 0.01f32..10.0f32,
    ) {
        let reference = dequantize_i8_scalar(&input, scale).unwrap();
        let caps = cpu_capabilities();
        if caps.avx2 {
            prop_assert_eq!(&reference, &dequantize_i8_avx2(&input, scale).unwrap());
        }
        if caps.neon {
            prop_assert_eq!(&reference, &dequantize_i8_neon(&input, scale).unwrap());
        }
    }

    #[test]
    fn matmul_all_backends_agree(
        m in 1usize..12, k in 1usize..12, n in 1usize..12,
        seed in any::<u64>(),
    ) {
        let count = m * k + k * n;
        let mut state = seed | 1;
        let mut vals = Vec::with_capacity(count);
        for _ in 0..count {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            vals.push((((state >> 33) as f64 / (1u64 << 31) as f64) * 4.0 - 2.0) as f32);
        }
        let (a, b) = vals.split_at(m * k);
        let reference = matrix_multiply_scalar(a, b, m, k, n).unwrap();
        let caps = cpu_capabilities();
        if caps.avx2 && caps.fma {
            let got = matrix_multiply_avx2_fma(a, b, m, k, n).unwrap();
            for i in 0..m * n {
                prop_assert!((got[i] - reference[i]).abs() <= 1e-3 * reference[i].abs().max(1.0));
            }
        }
        if caps.neon {
            let got = matrix_multiply_neon(a, b, m, k, n).unwrap();
            for i in 0..m * n {
                prop_assert!((got[i] - reference[i]).abs() <= 1e-3 * reference[i].abs().max(1.0));
            }
        }
    }
}
