//! Typed errors for Barq-WASM.
//!
//! Unfinished or unavailable functionality must surface one of these variants
//! instead of panicking or silently returning success.

use thiserror::Error;

/// Errors returned by Barq-WASM public APIs.
#[derive(Debug, Error)]
pub enum BarqError {
    /// The requested capability is not implemented in this build.
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// A runtime operation was attempted before a runtime was initialized.
    #[error("runtime not initialized: {0}")]
    RuntimeNotInitialized(String),

    /// Pattern-based specialization was requested but is unavailable.
    #[error("specialization unavailable: {0}")]
    SpecializationUnavailable(String),

    /// The provided arguments are invalid for the requested operation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The byte stream is not a valid WebAssembly module.
    #[error("module validation failed: {0}")]
    Validation(String),

    /// The module could not be instantiated (bad imports, start trap, ...).
    #[error("instantiation failed: {0}")]
    Instantiation(String),

    /// No module has been loaded into the runtime yet.
    #[error("no module loaded: {0}")]
    ModuleNotLoaded(String),

    /// The requested export does not exist or has the wrong kind/type.
    #[error("missing or mismatched export '{name}': {reason}")]
    MissingExport { name: String, reason: String },

    /// Guest code trapped during execution.
    #[error("wasm trap: {0}")]
    Trap(String),

    /// Execution ran out of fuel.
    #[error("fuel exhausted after {consumed} units")]
    FuelExhausted { consumed: u64 },

    /// Execution exceeded the configured wall-clock deadline.
    #[error("execution timed out after {millis} ms")]
    Timeout { millis: u64 },

    /// Linear memory access outside the memory bounds, or no memory exported.
    #[error("memory access error: {0}")]
    MemoryAccess(String),

    /// Filesystem or other I/O failure.
    #[error("i/o error: {0}")]
    Io(String),
}

/// Convenience result alias for Barq-WASM APIs.
pub type BarqResult<T> = Result<T, BarqError>;
