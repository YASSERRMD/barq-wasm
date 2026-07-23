//! Truth-baseline tests: the renamed scalar kernels compute correct results
//! against naive scalar references.

#![cfg(not(target_arch = "wasm32"))]

use barq_wasm::wasm_bindings::*;

// ---------------------------------------------------------------------------
// Kernel correctness: unrolled scalar vs naive scalar references
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random f32 values in [-100, 100].
fn pseudo_random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 33) as f64) / ((1u64 << 31) as f64); // [0, 1)
            ((unit * 200.0) - 100.0) as f32
        })
        .collect()
}

fn assert_close(a: f32, b: f32, max_relative: f32, context: &str) {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        diff <= max_relative * scale,
        "{context}: {a} vs {b} (diff {diff})"
    );
}

#[test]
fn dot_product_unrolled_matches_scalar() {
    for &len in &[0usize, 1, 15, 16, 17, 127, 128, 129, 1024, 10_000] {
        let a = pseudo_random_f32(len, 1);
        let b = pseudo_random_f32(len, 2);
        let reference = dot_product_scalar(&a, &b);
        let unrolled = dot_product_unrolled_scalar(&a, &b);
        assert_close(reference, unrolled, 1e-4, &format!("dot product len={len}"));
    }
}

#[test]
fn vector_norm_unrolled_matches_scalar() {
    for &len in &[0usize, 1, 7, 8, 9, 1024, 10_000] {
        let a = pseudo_random_f32(len, 3);
        let reference = vector_norm_scalar(&a);
        let unrolled = vector_norm_unrolled_scalar(&a);
        assert_close(reference, unrolled, 1e-5, &format!("l2 norm len={len}"));
    }
}

#[test]
fn cosine_similarity_unrolled_matches_scalar() {
    for &len in &[1usize, 16, 100, 1000] {
        let a = pseudo_random_f32(len, 4);
        let b = pseudo_random_f32(len, 5);
        let reference = cosine_similarity_scalar(&a, &b);
        let unrolled = cosine_similarity_unrolled_scalar(&a, &b);
        assert_close(reference, unrolled, 1e-4, &format!("cosine len={len}"));
    }
}

#[test]
fn cosine_similarity_zero_vector_is_zero() {
    let zeros = vec![0.0f32; 64];
    let other = pseudo_random_f32(64, 6);
    assert_eq!(cosine_similarity_unrolled_scalar(&zeros, &other), 0.0);
}

#[test]
fn quantize_unrolled_matches_scalar() {
    for &len in &[0usize, 1, 4, 15, 16, 17, 1024, 10_000] {
        let input = pseudo_random_f32(len, 7);
        let reference = quantize_int8_scalar(&input, 0.75);
        let unrolled = quantize_int8_unrolled_scalar(&input, 0.75);
        assert_eq!(reference.len(), unrolled.len());
        for (i, (r, u)) in reference.iter().zip(unrolled.iter()).enumerate() {
            let delta = (i32::from(*r) - i32::from(*u)).abs();
            assert!(
                delta <= 1,
                "quantize len={len} idx={i}: scalar {r} vs unrolled {u}"
            );
        }
    }
}

#[test]
fn quantize_saturates_to_i8_range() {
    let input = vec![1e6f32, -1e6, 400.0, -400.0];
    let q = quantize_int8_unrolled_scalar(&input, 1.0);
    assert_eq!(q, vec![127i8, -128, 127, -128]);
}

#[test]
fn matrix_multiply_tiled_matches_scalar() {
    for &n in &[1usize, 2, 16, 31, 32, 33, 64] {
        let a = pseudo_random_f32(n * n, 8);
        let b = pseudo_random_f32(n * n, 9);
        let reference = matrix_multiply_scalar(&a, &b, n);
        let tiled = matrix_multiply_tiled(&a, &b, n);
        for i in 0..n * n {
            assert_close(
                reference[i],
                tiled[i],
                1e-3,
                &format!("matmul n={n} idx={i}"),
            );
        }
    }
}

#[test]
fn lz4_experimental_small_input_is_verbatim_copy() {
    // Documented behavior: below 128 KiB the function performs no compression.
    let input: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
    assert_eq!(lz4_compress_experimental(&input), input);
}

#[test]
fn buffer_copy_baseline_is_identity() {
    let input: Vec<u8> = (0..255u8).collect();
    assert_eq!(buffer_copy_baseline(&input), input);
}
