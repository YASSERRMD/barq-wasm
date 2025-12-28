pub struct MemoryPool {
    // Placeholder
    _capacity: usize,
}

impl MemoryPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            _capacity: capacity,
        }
    }

    pub fn allocate(&self, _size: usize) {
        todo!("Implement allocate")
    }
}
