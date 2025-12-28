pub struct BenchmarkResult {
    pub name: String,
    pub duration_ns: u64,
}

impl BenchmarkResult {
    pub fn new(name: &str, duration_ns: u64) -> Self {
        Self {
            name: name.to_string(),
            duration_ns,
        }
    }
}
