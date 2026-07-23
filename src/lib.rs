#[cfg(feature = "analyzer")]
pub mod analyzer;
#[cfg(feature = "bench-tool")]
pub mod bench;
pub mod error;
#[cfg(feature = "jit-specialization")]
pub mod jit;
pub mod kernels;
#[cfg(feature = "native-runtime")]
pub mod runtime;
pub mod wasm_bindings;

pub use error::{BarqError, BarqResult};
// Re-export compute kernels for easy access
pub use wasm_bindings::*;
