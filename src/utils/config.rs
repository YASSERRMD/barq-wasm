use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub max_memory_mb: usize,
    pub enable_profiling: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_memory_mb: 512,
            enable_profiling: false,
        }
    }
}
