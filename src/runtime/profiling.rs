pub struct ExecutionProfiler {
    // Placeholder
    _data: (),
}

impl Default for ExecutionProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionProfiler {
    pub fn new() -> Self {
        Self { _data: () }
    }

    pub fn start(&self) {
        todo!("Implement start")
    }
}
