//! WebAssembly (wasm32) kernels.
//!
//! The scalar references in [`crate::kernels::scalar`] are target-independent
//! and serve as the ground truth on wasm32 too. This module adds explicit
//! `simd128` implementations, compiled only when the `simd128` target feature
//! is enabled (the "simd" browser bundle). The scalar bundle simply does not
//! contain them — nothing is silently substituted.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod simd128;
