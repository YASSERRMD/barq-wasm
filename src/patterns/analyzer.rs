use crate::patterns::{ai, compression, database, vectors, PatternProfile};
use anyhow::Result;

pub struct PatternAnalyzer;

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, bytecode: &[u8]) -> Result<PatternProfile> {
        // Run all detectors
        let comp_lz4 = compression::detect_lz4_decompression(bytecode);
        let comp_zstd = compression::detect_zstd_decompression(bytecode);
        let comp_brotli = compression::detect_brotli_decompression(bytecode);

        let best_comp = if comp_lz4.confidence >= comp_zstd.confidence
            && comp_lz4.confidence >= comp_brotli.confidence
        {
            comp_lz4
        } else if comp_zstd.confidence >= comp_brotli.confidence {
            comp_zstd
        } else {
            comp_brotli
        };

        let vec_matmul = vectors::detect_matrix_multiply(bytecode);
        let vec_dot = vectors::detect_dot_product(bytecode);
        let vec_norm = vectors::detect_vector_norm(bytecode);
        let vec_cosine = vectors::detect_cosine_similarity(bytecode);

        let best_vec = [vec_matmul, vec_dot, vec_norm, vec_cosine]
            .into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        let db_mongo = database::detect_mongodb_pattern(bytecode);
        let db_file = database::detect_filenet_pattern(bytecode);
        let db_io = database::detect_file_io_pattern(bytecode);
        let db_net = database::detect_network_io_pattern(bytecode);

        let best_db = [db_mongo, db_file, db_io, db_net]
            .into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        let ai_quant = ai::detect_quantization(bytecode);
        let ai_mv = ai::detect_matrix_vector_multiplication(bytecode);
        let ai_conv = ai::detect_convolution(bytecode);
        let ai_soft = ai::detect_softmax(bytecode);

        let best_ai = [ai_quant, ai_mv, ai_conv, ai_soft]
            .into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        // Determine primary pattern
        let patterns = [
            ("Compression", best_comp.confidence),
            ("Vector", best_vec.confidence),
            ("Database", best_db.confidence),
            ("AI", best_ai.confidence),
        ];

        let (category, max_conf) = patterns
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let primary_pattern = if max_conf > 0.5 {
            match category {
                "Compression" => best_comp.pattern.clone(),
                "Vector" => best_vec.pattern.clone(),
                "Database" => best_db.pattern.clone(),
                "AI" => best_ai.pattern.clone(),
                _ => "Generic".to_string(),
            }
        } else {
            "Generic Code".to_string()
        };

        let profile = PatternProfile {
            compression: best_comp,
            vector: best_vec,
            database: best_db,
            ai: best_ai,
            primary_pattern: primary_pattern.clone(),
            optimization_suggestion: self.suggest_optimization_internal(&primary_pattern),
        };

        Ok(profile)
    }

    pub fn suggest_optimization(&self, profile: &PatternProfile) -> String {
        profile.optimization_suggestion.clone()
    }

    fn suggest_optimization_internal(&self, primary_pattern: &str) -> String {
        match primary_pattern {
            p if p.contains("LZ4") => "Use specialized LZ4 decompressor syscall".to_string(),
            p if p.contains("Matrix Multiply") => "Enable AVX-512 SIMD vectorization".to_string(),
            p if p.contains("MongoDB") => "Bypass WASM heap for BSON serialization".to_string(),
            p if p.contains("Quantization") => {
                "Use INT8 tensor specialized instructions".to_string()
            }
            "Generic Code" => "Standard Cranelift optimization suite".to_string(),
            _ => format!("Apply {} specialized optimization", primary_pattern),
        }
    }
}
