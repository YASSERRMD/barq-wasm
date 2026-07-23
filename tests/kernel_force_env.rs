//! BARQ_FORCE_KERNEL override behavior.
//!
//! Lives in its own test binary because it mutates process-global
//! environment variables; cargo runs separate test binaries serially, so no
//! other kernel test can observe the override mid-flight.

#![cfg(not(target_arch = "wasm32"))]

use barq_wasm::error::BarqError;
use barq_wasm::kernels::*;

#[test]
fn forced_kernel_env_behavior() {
    // All env manipulation lives in this single test to avoid races between
    // parallel tests. select_backend() reads the variable per call.
    let caps = cpu_capabilities();
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
    let reference = dot_product_scalar(&a, &b).unwrap();

    std::env::set_var("BARQ_FORCE_KERNEL", "scalar");
    let exec = dot_product_ex(&a, &b).unwrap();
    assert_eq!(exec.backend, KernelBackend::Scalar);
    assert_eq!(exec.value, reference);

    std::env::set_var("BARQ_FORCE_KERNEL", "definitely-not-a-backend");
    assert!(matches!(
        dot_product(&a, &b),
        Err(BarqError::InvalidArgument(_))
    ));

    // Force each SIMD backend: supported -> runs it; unsupported -> typed error.
    for (name, backend, supported) in [
        ("avx2", KernelBackend::Avx2, caps.avx2),
        ("avx2-fma", KernelBackend::Avx2Fma, caps.avx2 && caps.fma),
        ("neon", KernelBackend::Neon, caps.neon),
    ] {
        std::env::set_var("BARQ_FORCE_KERNEL", name);
        match dot_product_ex(&a, &b) {
            Ok(exec) => {
                assert!(supported, "{name} ran but is not supported");
                assert_eq!(exec.backend, backend);
                assert_eq!(exec.value, reference);
            }
            Err(BarqError::UnsupportedFeature(_)) => {
                assert!(!supported, "{name} is supported but was refused");
            }
            Err(other) => panic!("unexpected error for {name}: {other:?}"),
        }
    }

    std::env::remove_var("BARQ_FORCE_KERNEL");
}
