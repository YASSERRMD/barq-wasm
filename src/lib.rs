pub mod analyzer;
pub mod codegen;
pub mod executor;
pub mod patterns;
pub mod runtime;
pub mod syscalls;
pub mod utils;
pub mod wasm_bindings;

// Re-export wasm bindings for easy access
pub use wasm_bindings::*;
