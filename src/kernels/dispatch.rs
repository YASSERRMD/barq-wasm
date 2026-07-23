//! Runtime CPU feature detection and backend selection.
//!
//! Detection uses `is_x86_feature_detected!` / `is_aarch64_feature_detected!`
//! and is cached once. `BARQ_FORCE_KERNEL` can force a backend for testing;
//! forcing an unsupported backend returns a typed error rather than executing
//! illegal instructions.

use crate::error::{BarqError, BarqResult};
use std::sync::OnceLock;

/// CPU architecture of the running process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Other,
}

/// Detected CPU capabilities (cached).
#[derive(Debug, Clone, Copy)]
pub struct CpuCapabilities {
    pub architecture: Architecture,
    pub sse42: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub neon: bool,
}

/// Which kernel implementation actually executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    Scalar,
    Avx2,
    Avx2Fma,
    Neon,
}

impl std::fmt::Display for KernelBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelBackend::Scalar => write!(f, "scalar"),
            KernelBackend::Avx2 => write!(f, "avx2"),
            KernelBackend::Avx2Fma => write!(f, "avx2+fma"),
            KernelBackend::Neon => write!(f, "neon"),
        }
    }
}

/// A kernel result together with the backend that produced it, so tests and
/// callers can prove which implementation ran.
#[derive(Debug, Clone, Copy)]
pub struct KernelExecution<T> {
    pub value: T,
    pub backend: KernelBackend,
}

/// Detect CPU capabilities once.
pub fn cpu_capabilities() -> &'static CpuCapabilities {
    static CAPS: OnceLock<CpuCapabilities> = OnceLock::new();
    CAPS.get_or_init(detect)
}

#[cfg(target_arch = "x86_64")]
fn detect() -> CpuCapabilities {
    CpuCapabilities {
        architecture: Architecture::X86_64,
        sse42: std::arch::is_x86_feature_detected!("sse4.2"),
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        neon: false,
    }
}

#[cfg(target_arch = "aarch64")]
fn detect() -> CpuCapabilities {
    CpuCapabilities {
        architecture: Architecture::Aarch64,
        sse42: false,
        avx2: false,
        fma: false,
        avx512f: false,
        neon: std::arch::is_aarch64_feature_detected!("neon"),
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> CpuCapabilities {
    CpuCapabilities {
        architecture: Architecture::Other,
        sse42: false,
        avx2: false,
        fma: false,
        avx512f: false,
        neon: false,
    }
}

/// Select the backend for this invocation.
///
/// Honors `BARQ_FORCE_KERNEL` (`scalar`, `avx2`, `avx2-fma`, `neon`), read on
/// every call so tests can vary it. Without an override, picks the best
/// backend the detected CPU supports.
pub fn select_backend() -> BarqResult<KernelBackend> {
    let caps = cpu_capabilities();
    match std::env::var("BARQ_FORCE_KERNEL") {
        Ok(forced) => {
            let backend = match forced.as_str() {
                "scalar" => KernelBackend::Scalar,
                "avx2" => KernelBackend::Avx2,
                "avx2-fma" => KernelBackend::Avx2Fma,
                "neon" => KernelBackend::Neon,
                other => {
                    return Err(BarqError::InvalidArgument(format!(
                        "BARQ_FORCE_KERNEL: unknown backend '{other}' \
                         (expected scalar|avx2|avx2-fma|neon)"
                    )))
                }
            };
            if backend_supported(backend, caps) {
                Ok(backend)
            } else {
                Err(BarqError::UnsupportedFeature(format!(
                    "BARQ_FORCE_KERNEL={forced} but this CPU does not support it \
                     (caps: {caps:?})"
                )))
            }
        }
        Err(_) => Ok(best_backend(caps)),
    }
}

pub(crate) fn backend_supported(backend: KernelBackend, caps: &CpuCapabilities) -> bool {
    match backend {
        KernelBackend::Scalar => true,
        KernelBackend::Avx2 => caps.avx2,
        KernelBackend::Avx2Fma => caps.avx2 && caps.fma,
        KernelBackend::Neon => caps.neon,
    }
}

fn best_backend(caps: &CpuCapabilities) -> KernelBackend {
    if caps.avx2 && caps.fma {
        KernelBackend::Avx2Fma
    } else if caps.avx2 {
        KernelBackend::Avx2
    } else if caps.neon {
        KernelBackend::Neon
    } else {
        KernelBackend::Scalar
    }
}
