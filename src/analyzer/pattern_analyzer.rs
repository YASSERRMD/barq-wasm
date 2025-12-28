use crate::analyzer::{ai_patterns, compression_patterns, database_patterns, vector_patterns};

#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Compression,
    Vector,
    Database,
    AI,
}

#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub name: String,
    pub pattern_type: PatternType,
    pub confidence: f32,
    pub optimization_hint: String,
}

pub struct PatternAnalyzer {
    confidence_threshold: f32,
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new(0.75)
    }
}

impl PatternAnalyzer {
    pub fn new(threshold: f32) -> Self {
        Self {
            confidence_threshold: threshold,
        }
    }

    pub fn analyze(&self, bytecode: &[u8]) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        // Compression
        if let Some(m) = compression_patterns::detect_lz4(bytecode) {
            matches.push(m);
        }
        if let Some(m) = compression_patterns::detect_zstd(bytecode) {
            matches.push(m);
        }
        if let Some(m) = compression_patterns::detect_brotli(bytecode) {
            matches.push(m);
        }

        // Vector
        if let Some(m) = vector_patterns::detect_matrix_multiply(bytecode) {
            matches.push(m);
        }
        if let Some(m) = vector_patterns::detect_dot_product(bytecode) {
            matches.push(m);
        }
        if let Some(m) = vector_patterns::detect_vector_norm(bytecode) {
            matches.push(m);
        }
        if let Some(m) = vector_patterns::detect_cosine_similarity(bytecode) {
            matches.push(m);
        }
        if let Some(m) = vector_patterns::detect_matrix_transpose(bytecode) {
            matches.push(m);
        }

        // Database
        if let Some(m) = database_patterns::detect_mongodb(bytecode) {
            matches.push(m);
        }
        if let Some(m) = database_patterns::detect_filenet(bytecode) {
            matches.push(m);
        }

        // AI
        if let Some(m) = ai_patterns::detect_int8_quantization(bytecode) {
            matches.push(m);
        }
        if let Some(m) = ai_patterns::detect_convolution_layer(bytecode) {
            matches.push(m);
        }
        if let Some(m) = ai_patterns::detect_attention_layer(bytecode) {
            matches.push(m);
        }

        // Filter and Sort
        matches.retain(|m| m.confidence >= self.confidence_threshold);
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }
}
