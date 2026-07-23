#![cfg(all(not(target_arch = "wasm32"), feature = "jit-specialization"))]

//! Cranelift JIT tests: compile complete functions, execute them, and verify
//! outputs against scalar references. A test asserting only that machine
//! code is non-empty would be worthless and is deliberately absent.

use barq_wasm::error::BarqError;
use barq_wasm::jit::{JitAddI64, JitDotProduct};
use barq_wasm::kernels;

fn pseudo_random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 33) as f64 / (1u64 << 31) as f64) * 10.0 - 5.0) as f32
        })
        .collect()
}

#[test]
fn jit_add_computes_real_sums() {
    let add = JitAddI64::compile().expect("compile");
    assert_eq!(add.call(20, 22), 42);
    assert_eq!(add.call(-5, 5), 0);
    assert_eq!(add.call(i64::MAX, 0), i64::MAX);
    for i in -100i64..100 {
        assert_eq!(add.call(i, i * 3), i * 4);
    }
}

#[test]
fn jit_dot_product_matches_scalar_reference() {
    let jit = JitDotProduct::compile().expect("compile");
    for len in [0usize, 1, 2, 7, 8, 9, 63, 64, 65, 1000, 10_007] {
        let a = pseudo_random_f32(len, 21);
        let b = pseudo_random_f32(len, 22);
        let reference = kernels::dot_product_scalar(&a, &b).unwrap();
        let got = jit.call(&a, &b).unwrap();
        let tol = 1e-4 * reference.abs().max(1.0);
        assert!(
            (reference - got).abs() <= tol,
            "len={len}: scalar {reference} vs jit {got}"
        );
    }
}

#[test]
fn jit_dot_product_randomized_lengths() {
    // Randomized mini-fuzz over lengths and seeds.
    let jit = JitDotProduct::compile().expect("compile");
    let mut state = 0xDEADBEEFu64;
    for _ in 0..200 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let len = (state >> 33) as usize % 300;
        let a = pseudo_random_f32(len, state);
        let b = pseudo_random_f32(len, state ^ 0xFFFF);
        let reference = kernels::dot_product_scalar(&a, &b).unwrap();
        let got = jit.call(&a, &b).unwrap();
        assert!(
            (reference - got).abs() <= 1e-4 * reference.abs().max(1.0),
            "len={len}"
        );
    }
}

#[test]
fn jit_dot_product_length_mismatch_is_typed_error() {
    let jit = JitDotProduct::compile().expect("compile");
    let err = jit.call(&[1.0, 2.0], &[1.0]).unwrap_err();
    assert!(matches!(err, BarqError::InvalidArgument(_)));
}

#[test]
fn jit_multiple_instances_coexist_and_free_safely() {
    let first = JitDotProduct::compile().expect("compile");
    let second = JitDotProduct::compile().expect("compile");
    let a = pseudo_random_f32(128, 31);
    let b = pseudo_random_f32(128, 32);
    let r1 = first.call(&a, &b).unwrap();
    let r2 = second.call(&a, &b).unwrap();
    assert_eq!(r1, r2);
    drop(first);
    // Second instance must still work after the first frees its memory.
    let r3 = second.call(&a, &b).unwrap();
    assert_eq!(r2, r3);
}
