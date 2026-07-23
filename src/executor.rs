//! Native execution entry point.
//!
//! No real WebAssembly runtime exists in this build yet. Construction fails
//! with a typed error so callers (including the CLI) cannot report success
//! without executing anything. A Wasmtime-backed runtime is Phase 2 scope.

use crate::error::{BarqError, BarqResult};

pub struct BarqRuntime {
    // Intentionally not constructible until a real runtime exists.
    _private: (),
}

impl BarqRuntime {
    /// Attempt to create a native runtime.
    ///
    /// Always returns `BarqError::UnsupportedFeature` in this build: there is
    /// no module loading, validation, instantiation, or execution yet.
    pub fn new() -> BarqResult<Self> {
        Err(BarqError::UnsupportedFeature(
            "native WebAssembly execution is not implemented in this build; \
             see docs/implementation-inventory.md"
                .to_string(),
        ))
    }

    /// Run a loaded module. Unreachable while `new()` returns an error.
    pub fn run(&self) -> BarqResult<()> {
        Err(BarqError::RuntimeNotInitialized(
            "no module has been loaded; module execution is not implemented".to_string(),
        ))
    }
}
