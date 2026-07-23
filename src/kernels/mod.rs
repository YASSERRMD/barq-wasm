//! Compute kernels: scalar references plus explicit native SIMD
//! implementations with runtime CPU detection and truthful dispatch.
//!
//! Naming policy:
//! - `*_scalar` — naive scalar reference (ground truth).
//! - `*_avx2`, `*_avx2_fma`, `*_neon` — explicit SIMD; safe wrappers that
//!   return a typed error when the CPU lacks the feature.
//! - Bare names (e.g. [`dot_product`]) — dispatch to the best available
//!   backend; the `*_ex` variants also report which backend actually ran.

pub mod dispatch;
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod arm;
#[cfg(target_arch = "x86_64")]
pub mod x86;

pub use dispatch::{
    cpu_capabilities, select_backend, Architecture, CpuCapabilities, KernelBackend,
    KernelExecution,
};

use crate::error::{BarqError, BarqResult};
use scalar::check_equal_len;

fn feature_err(what: &str) -> BarqError {
    BarqError::UnsupportedFeature(format!(
        "{what} is not supported on this CPU (caps: {:?})",
        cpu_capabilities()
    ))
}

fn check_scale(scale: f32) -> BarqResult<()> {
    if scale == 0.0 || !scale.is_finite() {
        return Err(BarqError::InvalidArgument(format!(
            "quantization scale must be finite and non-zero, got {scale}"
        )));
    }
    Ok(())
}

fn check_matrix_dims(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> BarqResult<()> {
    if a.len() != m * k || b.len() != k * n {
        return Err(BarqError::InvalidArgument(format!(
            "matrix dims mismatch: a.len()={} (want {m}x{k}={}), b.len()={} (want {k}x{n}={})",
            a.len(),
            m * k,
            b.len(),
            k * n
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar reference wrappers
// ---------------------------------------------------------------------------

pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> BarqResult<f32> {
    check_equal_len(a, b)?;
    Ok(scalar::dot_product_f32(a, b))
}

pub fn l2_norm_scalar(a: &[f32]) -> BarqResult<f32> {
    Ok(scalar::l2_norm_f32(a))
}

pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> BarqResult<f32> {
    check_equal_len(a, b)?;
    Ok(scalar::cosine_similarity_f32(a, b))
}

pub fn quantize_i8_scalar(input: &[f32], scale: f32) -> BarqResult<Vec<i8>> {
    check_scale(scale)?;
    let mut out = Vec::new();
    scalar::quantize_f32_to_i8(input, scale, &mut out);
    Ok(out)
}

pub fn dequantize_i8_scalar(input: &[i8], scale: f32) -> BarqResult<Vec<f32>> {
    let mut out = Vec::new();
    scalar::dequantize_i8_to_f32(input, scale, &mut out);
    Ok(out)
}

pub fn matrix_multiply_scalar(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> BarqResult<Vec<f32>> {
    check_matrix_dims(a, b, m, k, n)?;
    Ok(scalar::matrix_multiply_f32(a, b, m, k, n))
}

pub fn conv2d_scalar(
    input: &[f32],
    w: usize,
    h: usize,
    kernel: &[f32],
    ksize: usize,
) -> BarqResult<Vec<f32>> {
    if input.len() != w * h || kernel.len() != ksize * ksize || ksize == 0 || ksize > w.min(h) {
        return Err(BarqError::InvalidArgument(
            "conv2d dimensions are inconsistent".to_string(),
        ));
    }
    Ok(scalar::conv2d_f32(input, w, h, kernel, ksize))
}

// ---------------------------------------------------------------------------
// AVX2 wrappers (x86-64 only): verified feature detection before dispatch
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2_wrappers {
    use super::*;

    fn require_avx2() -> BarqResult<()> {
        if cpu_capabilities().avx2 {
            Ok(())
        } else {
            Err(feature_err("AVX2"))
        }
    }

    fn require_avx2_fma() -> BarqResult<()> {
        let caps = cpu_capabilities();
        if caps.avx2 && caps.fma {
            Ok(())
        } else {
            Err(feature_err("AVX2+FMA"))
        }
    }

    pub fn dot_product_avx2(a: &[f32], b: &[f32]) -> BarqResult<f32> {
        check_equal_len(a, b)?;
        require_avx2()?;
        // SAFETY: AVX2 support verified above; lengths equal.
        Ok(unsafe { x86::avx2::dot_product_f32(a, b) })
    }

    pub fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> BarqResult<f32> {
        check_equal_len(a, b)?;
        require_avx2_fma()?;
        // SAFETY: AVX2+FMA support verified above; lengths equal.
        Ok(unsafe { x86::avx2::dot_product_f32_fma(a, b) })
    }

    pub fn l2_norm_avx2(a: &[f32]) -> BarqResult<f32> {
        require_avx2()?;
        // SAFETY: AVX2 support verified above.
        Ok(unsafe { x86::avx2::sum_of_squares_f32(a) }.sqrt())
    }

    pub fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> BarqResult<f32> {
        check_equal_len(a, b)?;
        require_avx2()?;
        // SAFETY: AVX2 support verified above; lengths equal.
        let (dot, na, nb) = unsafe {
            (
                x86::avx2::dot_product_f32(a, b),
                x86::avx2::sum_of_squares_f32(a).sqrt(),
                x86::avx2::sum_of_squares_f32(b).sqrt(),
            )
        };
        Ok(if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            0.0
        })
    }

    pub fn quantize_i8_avx2(input: &[f32], scale: f32) -> BarqResult<Vec<i8>> {
        check_scale(scale)?;
        require_avx2()?;
        let mut out = Vec::new();
        // SAFETY: AVX2 support verified above; scale checked.
        unsafe { x86::avx2::quantize_f32_to_i8(input, scale, &mut out) };
        Ok(out)
    }

    pub fn dequantize_i8_avx2(input: &[i8], scale: f32) -> BarqResult<Vec<f32>> {
        require_avx2()?;
        let mut out = Vec::new();
        // SAFETY: AVX2 support verified above.
        unsafe { x86::avx2::dequantize_i8_to_f32(input, scale, &mut out) };
        Ok(out)
    }

    pub fn matrix_multiply_avx2_fma(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BarqResult<Vec<f32>> {
        check_matrix_dims(a, b, m, k, n)?;
        require_avx2_fma()?;
        // SAFETY: AVX2+FMA support verified above; dims checked.
        Ok(unsafe { x86::avx2::matrix_multiply_f32_fma(a, b, m, k, n) })
    }
}

#[cfg(target_arch = "x86_64")]
pub use avx2_wrappers::*;

#[cfg(not(target_arch = "x86_64"))]
mod avx2_stubs {
    use super::*;

    pub fn dot_product_avx2(_a: &[f32], _b: &[f32]) -> BarqResult<f32> {
        Err(feature_err("AVX2"))
    }
    pub fn dot_product_avx2_fma(_a: &[f32], _b: &[f32]) -> BarqResult<f32> {
        Err(feature_err("AVX2+FMA"))
    }
    pub fn l2_norm_avx2(_a: &[f32]) -> BarqResult<f32> {
        Err(feature_err("AVX2"))
    }
    pub fn cosine_similarity_avx2(_a: &[f32], _b: &[f32]) -> BarqResult<f32> {
        Err(feature_err("AVX2"))
    }
    pub fn quantize_i8_avx2(_input: &[f32], _scale: f32) -> BarqResult<Vec<i8>> {
        Err(feature_err("AVX2"))
    }
    pub fn dequantize_i8_avx2(_input: &[i8], _scale: f32) -> BarqResult<Vec<f32>> {
        Err(feature_err("AVX2"))
    }
    pub fn matrix_multiply_avx2_fma(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> BarqResult<Vec<f32>> {
        Err(feature_err("AVX2+FMA"))
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub use avx2_stubs::*;

// ---------------------------------------------------------------------------
// NEON wrappers (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon_wrappers {
    use super::*;

    fn require_neon() -> BarqResult<()> {
        if cpu_capabilities().neon {
            Ok(())
        } else {
            Err(feature_err("NEON"))
        }
    }

    pub fn dot_product_neon(a: &[f32], b: &[f32]) -> BarqResult<f32> {
        check_equal_len(a, b)?;
        require_neon()?;
        // SAFETY: NEON support verified above; lengths equal.
        Ok(unsafe { arm::neon::dot_product_f32(a, b) })
    }

    pub fn l2_norm_neon(a: &[f32]) -> BarqResult<f32> {
        require_neon()?;
        // SAFETY: NEON support verified above.
        Ok(unsafe { arm::neon::sum_of_squares_f32(a) }.sqrt())
    }

    pub fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> BarqResult<f32> {
        check_equal_len(a, b)?;
        require_neon()?;
        // SAFETY: NEON support verified above; lengths equal.
        let (dot, na, nb) = unsafe {
            (
                arm::neon::dot_product_f32(a, b),
                arm::neon::sum_of_squares_f32(a).sqrt(),
                arm::neon::sum_of_squares_f32(b).sqrt(),
            )
        };
        Ok(if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            0.0
        })
    }

    pub fn quantize_i8_neon(input: &[f32], scale: f32) -> BarqResult<Vec<i8>> {
        check_scale(scale)?;
        require_neon()?;
        let mut out = Vec::new();
        // SAFETY: NEON support verified above; scale checked.
        unsafe { arm::neon::quantize_f32_to_i8(input, scale, &mut out) };
        Ok(out)
    }

    pub fn dequantize_i8_neon(input: &[i8], scale: f32) -> BarqResult<Vec<f32>> {
        require_neon()?;
        let mut out = Vec::new();
        // SAFETY: NEON support verified above.
        unsafe { arm::neon::dequantize_i8_to_f32(input, scale, &mut out) };
        Ok(out)
    }

    pub fn matrix_multiply_neon(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BarqResult<Vec<f32>> {
        check_matrix_dims(a, b, m, k, n)?;
        require_neon()?;
        // SAFETY: NEON support verified above; dims checked.
        Ok(unsafe { arm::neon::matrix_multiply_f32(a, b, m, k, n) })
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon_wrappers::*;

#[cfg(not(target_arch = "aarch64"))]
mod neon_stubs {
    use super::*;

    pub fn dot_product_neon(_a: &[f32], _b: &[f32]) -> BarqResult<f32> {
        Err(feature_err("NEON"))
    }
    pub fn l2_norm_neon(_a: &[f32]) -> BarqResult<f32> {
        Err(feature_err("NEON"))
    }
    pub fn cosine_similarity_neon(_a: &[f32], _b: &[f32]) -> BarqResult<f32> {
        Err(feature_err("NEON"))
    }
    pub fn quantize_i8_neon(_input: &[f32], _scale: f32) -> BarqResult<Vec<i8>> {
        Err(feature_err("NEON"))
    }
    pub fn dequantize_i8_neon(_input: &[i8], _scale: f32) -> BarqResult<Vec<f32>> {
        Err(feature_err("NEON"))
    }
    pub fn matrix_multiply_neon(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> BarqResult<Vec<f32>> {
        Err(feature_err("NEON"))
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub use neon_stubs::*;

// ---------------------------------------------------------------------------
// Auto-dispatch entry points
// ---------------------------------------------------------------------------

macro_rules! dispatched {
    ($backend:expr, scalar: $s:expr, avx2: $a:expr, avx2_fma: $af:expr, neon: $n:expr) => {
        match $backend {
            KernelBackend::Scalar => $s,
            KernelBackend::Avx2 => $a,
            KernelBackend::Avx2Fma => $af,
            KernelBackend::Neon => $n,
        }
    };
}

/// Dot product on the best available backend, reporting which one ran.
pub fn dot_product_ex(a: &[f32], b: &[f32]) -> BarqResult<KernelExecution<f32>> {
    let backend = select_backend()?;
    let value = dispatched!(backend,
        scalar: dot_product_scalar(a, b)?,
        avx2: dot_product_avx2(a, b)?,
        avx2_fma: dot_product_avx2_fma(a, b)?,
        neon: dot_product_neon(a, b)?);
    Ok(KernelExecution { value, backend })
}

/// Dot product on the best available backend.
pub fn dot_product(a: &[f32], b: &[f32]) -> BarqResult<f32> {
    Ok(dot_product_ex(a, b)?.value)
}

/// L2 norm on the best available backend, reporting which one ran.
pub fn l2_norm_ex(a: &[f32]) -> BarqResult<KernelExecution<f32>> {
    let backend = select_backend()?;
    let value = dispatched!(backend,
        scalar: l2_norm_scalar(a)?,
        avx2: l2_norm_avx2(a)?,
        avx2_fma: l2_norm_avx2(a)?,
        neon: l2_norm_neon(a)?);
    Ok(KernelExecution { value, backend })
}

/// L2 norm on the best available backend.
pub fn l2_norm(a: &[f32]) -> BarqResult<f32> {
    Ok(l2_norm_ex(a)?.value)
}

/// Cosine similarity on the best available backend, reporting which one ran.
pub fn cosine_similarity_ex(a: &[f32], b: &[f32]) -> BarqResult<KernelExecution<f32>> {
    let backend = select_backend()?;
    let value = dispatched!(backend,
        scalar: cosine_similarity_scalar(a, b)?,
        avx2: cosine_similarity_avx2(a, b)?,
        avx2_fma: cosine_similarity_avx2(a, b)?,
        neon: cosine_similarity_neon(a, b)?);
    Ok(KernelExecution { value, backend })
}

/// Cosine similarity on the best available backend.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> BarqResult<f32> {
    Ok(cosine_similarity_ex(a, b)?.value)
}

/// INT8 quantization on the best available backend, reporting which one ran.
pub fn quantize_i8_ex(input: &[f32], scale: f32) -> BarqResult<KernelExecution<Vec<i8>>> {
    let backend = select_backend()?;
    let value = dispatched!(backend,
        scalar: quantize_i8_scalar(input, scale)?,
        avx2: quantize_i8_avx2(input, scale)?,
        avx2_fma: quantize_i8_avx2(input, scale)?,
        neon: quantize_i8_neon(input, scale)?);
    Ok(KernelExecution { value, backend })
}

/// INT8 quantization on the best available backend.
pub fn quantize_i8(input: &[f32], scale: f32) -> BarqResult<Vec<i8>> {
    Ok(quantize_i8_ex(input, scale)?.value)
}

/// INT8 dequantization on the best available backend, reporting which one ran.
pub fn dequantize_i8_ex(input: &[i8], scale: f32) -> BarqResult<KernelExecution<Vec<f32>>> {
    let backend = select_backend()?;
    let value = dispatched!(backend,
        scalar: dequantize_i8_scalar(input, scale)?,
        avx2: dequantize_i8_avx2(input, scale)?,
        avx2_fma: dequantize_i8_avx2(input, scale)?,
        neon: dequantize_i8_neon(input, scale)?);
    Ok(KernelExecution { value, backend })
}

/// INT8 dequantization on the best available backend.
pub fn dequantize_i8(input: &[i8], scale: f32) -> BarqResult<Vec<f32>> {
    Ok(dequantize_i8_ex(input, scale)?.value)
}

/// Matrix multiply on the best available backend, reporting which one ran.
///
/// Plain AVX2 without FMA falls back to the scalar reference: no non-FMA
/// AVX2 matmul kernel is implemented, and this dispatcher never mislabels
/// what actually ran.
pub fn matrix_multiply_ex(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> BarqResult<KernelExecution<Vec<f32>>> {
    let backend = select_backend()?;
    let (value, backend) = match backend {
        KernelBackend::Avx2Fma => (matrix_multiply_avx2_fma(a, b, m, k, n)?, backend),
        KernelBackend::Neon => (matrix_multiply_neon(a, b, m, k, n)?, backend),
        KernelBackend::Scalar | KernelBackend::Avx2 => (
            matrix_multiply_scalar(a, b, m, k, n)?,
            KernelBackend::Scalar,
        ),
    };
    Ok(KernelExecution { value, backend })
}

/// Matrix multiply on the best available backend.
pub fn matrix_multiply(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> BarqResult<Vec<f32>> {
    Ok(matrix_multiply_ex(a, b, m, k, n)?.value)
}

/// 2D convolution. Currently scalar-only on every architecture; the
/// execution report says so truthfully.
pub fn conv2d_ex(
    input: &[f32],
    w: usize,
    h: usize,
    kernel: &[f32],
    ksize: usize,
) -> BarqResult<KernelExecution<Vec<f32>>> {
    let value = conv2d_scalar(input, w, h, kernel, ksize)?;
    Ok(KernelExecution {
        value,
        backend: KernelBackend::Scalar,
    })
}

/// 2D convolution (scalar implementation).
pub fn conv2d(
    input: &[f32],
    w: usize,
    h: usize,
    kernel: &[f32],
    ksize: usize,
) -> BarqResult<Vec<f32>> {
    Ok(conv2d_ex(input, w, h, kernel, ksize)?.value)
}
