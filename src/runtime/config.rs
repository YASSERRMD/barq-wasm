//! Runtime configuration.

use std::time::Duration;

/// Configuration for a [`crate::runtime::Runtime`].
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Link WASI preview1 imports (`wasi_snapshot_preview1`).
    pub enable_wasi: bool,
    /// Inherit the host's stdout/stderr for WASI programs. When false, WASI
    /// output is captured in memory and retrievable via
    /// [`crate::runtime::Runtime::take_wasi_output`].
    pub inherit_stdio: bool,
    /// Argv passed to WASI programs (argv[0] is the program name).
    pub wasi_args: Vec<String>,
    /// Deterministic instruction budget. `None` disables fuel metering.
    pub fuel: Option<u64>,
    /// Wall-clock execution deadline per invocation, enforced via epoch
    /// interruption. `None` disables the deadline.
    pub timeout: Option<Duration>,
    /// Upper bound for guest linear memory, in bytes.
    pub max_memory_bytes: Option<usize>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enable_wasi: true,
            inherit_stdio: true,
            wasi_args: vec!["module".to_string()],
            fuel: None,
            timeout: None,
            max_memory_bytes: None,
        }
    }
}
