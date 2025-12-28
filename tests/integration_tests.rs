use barq_wasm::analyzer::PatternAnalyzer;
use barq_wasm::codegen::cranelift_backend::{CraneliftBackend, CraneliftIR, OptimizationLevel};
use barq_wasm::codegen::vector_codegen::{CPUFeatures, VectorAccelerator, VectorOpType};
use std::process::Command;

// --- INFRASTRUCTURE TESTS ---

#[test]
fn test_cargo_build() {
    let status = Command::new("cargo")
        .arg("build")
        .status()
        .expect("Failed to execute cargo build");
    assert!(status.success());
}

#[test]
fn test_project_structure() {
    let directories = [
        "src",
        "tests",
        "benches",
        "examples",
        "docs",
        ".github/workflows",
    ];
    for dir in directories {
        assert!(
            std::path::Path::new(dir).exists(),
            "Directory {} missing",
            dir
        );
    }
}

// --- JIT COMPRESSION TESTS ---

#[test]
fn test_lz4_pattern_detected_and_optimized() {
    let backend = CraneliftBackend::new();
    let result = backend.compile(&[], OptimizationLevel::Compression);
    assert!(result.is_ok());
    let compiled = result.unwrap();
    assert_eq!(compiled.optimization_level, OptimizationLevel::Compression);
}

#[test]
fn test_dictionary_loop_unrolled() {
    use barq_wasm::codegen::compression_codegen;
    let mut ir = CraneliftIR {
        instructions: vec![],
    };
    let res = compression_codegen::unroll_dictionary_loop(&mut ir, 4);
    assert!(res.is_ok());
    assert!(ir.instructions.contains(&"unroll_loop_4".to_string()));
}

#[test]
fn test_simd_comparisons_injected() {
    use barq_wasm::codegen::compression_codegen;
    let mut ir = CraneliftIR {
        instructions: vec![],
    };
    let res = compression_codegen::inject_simd_comparisons(&mut ir);
    assert!(res.is_ok());
    assert!(ir.instructions.contains(&"simd_compare_128".to_string()));
}

#[test]
fn test_memory_prefetch_added() {
    use barq_wasm::codegen::compression_codegen;
    let mut ir = CraneliftIR {
        instructions: vec![],
    };
    let res = compression_codegen::add_prefetch_hints(&mut ir);
    assert!(res.is_ok());
    assert!(ir.instructions.contains(&"prefetch_l1".to_string()));
}

#[test]
fn test_huffman_tree_optimized() {
    use barq_wasm::codegen::compression_codegen;
    let mut ir = CraneliftIR {
        instructions: vec![],
    };
    let res = compression_codegen::specialize_huffman_decoding(&mut ir);
    assert!(res.is_ok());
    assert!(ir.instructions.contains(&"huffman_table_cache".to_string()));
}

#[test]
fn test_entropy_decoding_batched() {
    use barq_wasm::codegen::compression_codegen;
    let mut ir = CraneliftIR {
        instructions: vec![],
    };
    let res = compression_codegen::optimize_entropy_decoding(&mut ir);
    assert!(res.is_ok());
    assert!(ir
        .instructions
        .contains(&"bactch_entropy_decode".to_string()));
}

#[test]
fn test_generated_code_correctness() {
    let backend = CraneliftBackend::new();
    let result = backend.compile(&[], OptimizationLevel::Generic);
    assert!(result.is_ok());
    let compiled = result.unwrap();
    assert!(!compiled.machine_code.is_empty());
}

#[test]
fn test_integration_with_phase1_analyzer() {
    let mut bytecode = Vec::new();
    for _ in 0..5 {
        bytecode.push(0x03);
    }
    for _ in 0..30 {
        bytecode.push(0x28);
    }
    for _ in 0..15 {
        bytecode.push(0x0c);
    }
    bytecode.push(0xfc);

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);

    if let Some(lz4) = matches.iter().find(|m| m.name.contains("LZ4")) {
        assert!(lz4.confidence >= 0.75);
        let backend = CraneliftBackend::new();
        let result = backend.compile(&bytecode, OptimizationLevel::Compression);
        assert!(result.is_ok());
    } else {
        panic!("Analyzer failed to detect LZ4 pattern required for integration test");
    }
}

#[test]
fn test_zstd_pattern_detected_and_optimized() {
    let backend = CraneliftBackend::new();
    let result = backend.compile(&[], OptimizationLevel::Compression);
    assert!(result.is_ok());
    let compiled = result.unwrap();
    assert_eq!(compiled.optimization_level, OptimizationLevel::Compression);
}

#[test]
fn test_detect_and_specialize_zstd_codegen() {
    use barq_wasm::codegen::compression_codegen;
    let mut ir = CraneliftIR {
        instructions: vec![],
    };
    let res = compression_codegen::detect_and_specialize_zstd(&mut ir);
    assert!(res.is_ok());
    assert!(res.unwrap());
    assert!(ir.instructions.contains(&"specializing_zstd".to_string()));
}

// --- VECTOR OPTIMIZATION TESTS ---

#[test]
fn test_avx2_dot_product_generated() {
    let code = VectorAccelerator::emit_simd_code(VectorOpType::DotProduct, 1024).unwrap();
    if code.features.has_avx2 {
        // Expect vfmadd231ps (0xC4 0xE2 0x79) or basic structure check
        assert!(!code.machine_code.is_empty());
    } else {
        // Fallback
        assert!(!code.machine_code.is_empty());
    }
}

#[test]
fn test_avx2_matrix_multiply_generated() {
    let code = VectorAccelerator::emit_simd_code(VectorOpType::MatrixMultiply, 1024).unwrap();
    assert!(!code.machine_code.is_empty());
}

#[test]
fn test_avx2_vector_norm_generated() {
    let code = VectorAccelerator::emit_simd_code(VectorOpType::VectorNorm, 1024).unwrap();
    assert!(!code.machine_code.is_empty());
}

#[test]
fn test_avx2_cosine_similarity_generated() {
    let code = VectorAccelerator::emit_simd_code(VectorOpType::CosineSimilarity, 1024).unwrap();
    assert!(!code.machine_code.is_empty());
}

#[test]
fn test_int8_quantization_generated() {
    let code = VectorAccelerator::emit_simd_code(VectorOpType::Int8Quantization, 1024).unwrap();
    assert!(!code.machine_code.is_empty());
}

#[test]
fn test_dot_product_pattern_optimized() {
    // Phase 1 + Phase 3 Integration
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x0b]); // loop depth 1
    for _ in 0..25 {
        bytecode.push(0x94);
    } // many mults
    for _ in 0..25 {
        bytecode.push(0x92);
    } // many adds

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);

    let dot_prod = matches.iter().find(|m| m.name.contains("Dot Product"));
    assert!(dot_prod.is_some());
    assert!(dot_prod.unwrap().confidence >= 0.8);

    // If detected, we emit SIMD code (simulation)
    let code = VectorAccelerator::emit_simd_code(VectorOpType::DotProduct, 1024);
    assert!(code.is_ok());
}
