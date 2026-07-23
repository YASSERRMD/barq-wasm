//! Browser (wasm32) tests, run with wasm-pack in a real headless browser:
//!
//!   wasm-pack test --headless --chrome  -- --no-default-features --features browser
//!   wasm-pack test --headless --firefox -- --no-default-features --features browser
//!
//! With RUSTFLAGS="-C target-feature=+simd128" the SIMD differential tests
//! run; without it, only the scalar tests compile — mirroring the two
//! shipped bundles.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn pseudo_random_f32(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed | 1;
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

#[wasm_bindgen_test]
fn scalar_dot_product_known_answer() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    assert_eq!(barq_wasm::wasm_bindings::dot_product_scalar(&a, &b), 70.0);
}

#[wasm_bindgen_test]
fn bundle_reports_its_simd_compilation_truthfully() {
    let enabled = barq_wasm::wasm_bindings::simd128_enabled();
    assert_eq!(enabled, cfg!(target_feature = "simd128"));
}

#[wasm_bindgen_test]
fn scalar_quantize_policy_holds_in_browser() {
    let input = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, 1.5, 2.5];
    let q = barq_wasm::kernels::quantize_i8_scalar(&input, 1.0).unwrap();
    assert_eq!(q, vec![0, 127, -128, 0, 2, 2]);
}

#[cfg(target_feature = "simd128")]
mod simd_tests {
    use super::*;
    use barq_wasm::kernels::scalar;
    use barq_wasm::kernels::wasm32::simd128;

    #[wasm_bindgen_test]
    fn simd_dot_product_matches_scalar() {
        for len in [0usize, 1, 3, 4, 5, 8, 9, 127, 1024, 100_000] {
            let a = pseudo_random_f32(len, 7, -100.0, 100.0);
            let b = pseudo_random_f32(len, 8, -100.0, 100.0);
            let reference = scalar::dot_product_f32(&a, &b);
            let got = simd128::dot_product_f32(&a, &b);
            let tol = 1e-4 * reference.abs().max(1.0);
            assert!(
                (reference - got).abs() <= tol,
                "len={len}: {reference} vs {got}"
            );
        }
    }

    #[wasm_bindgen_test]
    fn simd_l2_norm_and_cosine_match_scalar() {
        let a = pseudo_random_f32(1000, 9, -50.0, 50.0);
        let b = pseudo_random_f32(1000, 10, -50.0, 50.0);
        let n_ref = scalar::l2_norm_f32(&a);
        let n_got = simd128::l2_norm_f32(&a);
        assert!((n_ref - n_got).abs() <= 1e-4 * n_ref.max(1.0));
        let c_ref = scalar::cosine_similarity_f32(&a, &b);
        let c_got = simd128::cosine_similarity_f32(&a, &b);
        assert!((c_ref - c_got).abs() <= 1e-3);
    }

    #[wasm_bindgen_test]
    fn simd_quantize_bit_exact_including_special_values() {
        let mut input = pseudo_random_f32(1000, 11, -300.0, 300.0);
        input.extend_from_slice(&[
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.5,
            1.5,
            2.5,
            -0.5,
            -1.5,
            127.5,
            -128.5,
        ]);
        for scale in [1.0f32, 0.5, 0.037] {
            let mut reference = Vec::new();
            scalar::quantize_f32_to_i8(&input, scale, &mut reference);
            let mut got = Vec::new();
            simd128::quantize_f32_to_i8(&input, scale, &mut got);
            assert_eq!(reference, got, "scale={scale}");
        }
    }

    #[wasm_bindgen_test]
    fn simd_dequantize_bit_exact() {
        let input: Vec<i8> = (0..1000).map(|i| ((i * 37 + 11) % 256) as u8 as i8).collect();
        let mut reference = Vec::new();
        scalar::dequantize_i8_to_f32(&input, 0.125, &mut reference);
        let mut got = Vec::new();
        simd128::dequantize_i8_to_f32(&input, 0.125, &mut got);
        assert_eq!(reference, got);
    }

    #[wasm_bindgen_test]
    fn simd_elementwise_ops_match_scalar() {
        let a = pseudo_random_f32(1001, 12, -10.0, 10.0);
        let b = pseudo_random_f32(1001, 13, -10.0, 10.0);
        let mut sum = Vec::new();
        simd128::vector_add_f32(&a, &b, &mut sum);
        let mut prod = Vec::new();
        simd128::vector_mul_f32(&a, &b, &mut prod);
        for i in 0..a.len() {
            assert_eq!(sum[i], a[i] + b[i]);
            assert_eq!(prod[i], a[i] * b[i]);
        }
    }

    #[wasm_bindgen_test]
    fn simd_repeated_invocation_is_stable() {
        let a = pseudo_random_f32(4096, 14, -1.0, 1.0);
        let b = pseudo_random_f32(4096, 15, -1.0, 1.0);
        let first = simd128::dot_product_f32(&a, &b);
        for _ in 0..100 {
            assert_eq!(first, simd128::dot_product_f32(&a, &b));
        }
    }

    #[wasm_bindgen_test]
    fn simd_large_input_transfer() {
        // 500k elements exercises memory growth and large slice transfer.
        let a = pseudo_random_f32(500_000, 16, -1.0, 1.0);
        let b = pseudo_random_f32(500_000, 17, -1.0, 1.0);
        let reference = scalar::dot_product_f32(&a, &b);
        let got = simd128::dot_product_f32(&a, &b);
        assert!((reference - got).abs() <= 1e-2 * reference.abs().max(1.0));
    }
}
