//! Typed errors for Barq-WASM.
//!
//! Unfinished or unavailable functionality must surface one of these variants
//! instead of panicking (`todo!()`) or silently returning success.

use thiserror::Error;

/// Errors returned by Barq-WASM public APIs.
#[derive(Debug, Error)]
pub enum BarqError {
    /// The requested capability is not implemented in this build.
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// A runtime operation was attempted before a runtime was initialized,
    /// or no real runtime implementation exists in this build.
    #[error("runtime not initialized: {0}")]
    RuntimeNotInitialized(String),

    /// Pattern-based specialization was requested but is unavailable.
    #[error("specialization unavailable: {0}")]
    SpecializationUnavailable(String),

    /// The provided arguments are invalid for the requested operation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Convenience result alias for Barq-WASM APIs.
pub type BarqResult<T> = Result<T, BarqError>;
