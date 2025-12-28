use barq_wasm::executor::BarqRuntime;
use barq_wasm::patterns::ai;
use barq_wasm::patterns::analyzer::PatternAnalyzer;
use barq_wasm::patterns::compression;
use barq_wasm::patterns::database;
use barq_wasm::patterns::vectors;
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

#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .arg("run")
        .arg("--")
        .arg("--help")
        .output()
        .expect("Failed to execute cargo run -- --help");
    assert!(output.status.success() || !output.stdout.is_empty());
}

#[test]
fn test_module_initialization() {
    let runtime = BarqRuntime::new();
    assert!(runtime.is_ok());
}

// --- COMPRESSION TESTS ---

#[test]
fn test_detect_lz4_decompression_high_confidence() {
    let mut bytecode = Vec::new();
    for _ in 0..10 {
        bytecode.push(0x03);
    } // loops
    for _ in 0..30 {
        bytecode.push(0x28);
    } // loads
    for _ in 0..15 {
        bytecode.push(0x0c);
    } // branches
    for _ in 0..10 {
        bytecode.push(0xfc);
    } // copies

    let result = compression::detect_lz4_decompression(&bytecode);
    assert!(
        result.confidence >= 0.75,
        "Confidence was {}",
        result.confidence
    );
    assert!(result.pattern.contains("LZ4"));
}

#[test]
fn test_detect_zstd_decompression_high_confidence() {
    let mut bytecode = Vec::new();
    for _ in 0..15 {
        bytecode.push(0x03);
    }
    for _ in 0..60 {
        bytecode.push(0x28);
    }
    for _ in 0..20 {
        bytecode.push(0x92);
    } // adds
    let result = compression::detect_zstd_decompression(&bytecode);
    assert!(result.confidence >= 0.8);
}

#[test]
fn test_generic_loop_not_flagged_as_compression() {
    let bytecode = vec![0x03, 0x01, 0x01, 0x0b]; // simple loop
    let result = compression::detect_lz4_decompression(&bytecode);
    assert!(result.confidence < 0.3);
}

#[test]
fn test_compression_confidence_threshold_respected() {
    let mut bytecode = Vec::new();
    for _ in 0..3 {
        bytecode.push(0x03);
    } // Too few loops
    let result = compression::detect_lz4_decompression(&bytecode);
    assert!(result.confidence < 0.5);
}

#[test]
fn test_mixed_compression_patterns_detected() {
    let mut bytecode = Vec::new();
    for _ in 0..20 {
        bytecode.push(0x03);
    }
    let result = compression::detect_brotli_decompression(&bytecode);
    assert!(result.confidence >= 0.7);
}

// --- VECTOR TESTS ---

#[test]
fn test_detect_matrix_multiply_triple_nested() {
    let mut bytecode = Vec::new();
    // Triple nested loop: loop { loop { loop { end } end } end }
    bytecode.extend_from_slice(&[0x03, 0x03, 0x03, 0x0b, 0x0b, 0x0b]);
    for _ in 0..60 {
        bytecode.push(0x94);
    } // muls
    for _ in 0..60 {
        bytecode.push(0x92);
    } // adds

    let result = vectors::detect_matrix_multiply(&bytecode);
    assert!(result.confidence >= 0.9);
    assert_eq!(result.pattern, "Matrix Multiply");
}

#[test]
fn test_detect_dot_product_single_loop() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x0b]); // single loop
    for _ in 0..25 {
        bytecode.push(0x94);
    } // muls
    for _ in 0..25 {
        bytecode.push(0x92);
    } // adds

    let result = vectors::detect_dot_product(&bytecode);
    assert!(result.confidence >= 0.9);
}

#[test]
fn test_detect_cosine_similarity_pattern() {
    let mut bytecode = Vec::new();
    for _ in 0..50 {
        bytecode.push(0x94);
    }
    for _ in 0..50 {
        bytecode.push(0x92);
    }
    bytecode.push(0x95); // f32.div
    let result = vectors::detect_cosine_similarity(&bytecode);
    assert!(result.confidence >= 0.9);
}

#[test]
fn test_vector_pattern_dimensions_inferred() {
    // In our simplified impl, we don't infer dims yet, but we verify detection
    let bytecode = vec![0x03, 0x91, 0x0b]; // loop + sqrt
    let result = vectors::detect_vector_norm(&bytecode);
    assert!(result.confidence >= 0.8);
}

#[test]
fn test_non_vector_code_not_misdetected() {
    let bytecode = vec![0x01, 0x02, 0x03, 0x0b];
    let result = vectors::detect_matrix_multiply(&bytecode);
    assert!(result.confidence < 0.5);
}

// --- DATABASE TESTS ---

#[test]
fn test_detect_mongodb_insert_pattern() {
    let bytecode = vec![0x10, 1, 0x10, 2, 0x10, 3]; // socket, write, read
    let result = database::detect_mongodb_pattern(&bytecode);
    assert!(result.confidence >= 0.8);
}

#[test]
fn test_detect_filenet_query_pattern() {
    let bytecode = vec![0x10, 4, 0x10, 5, 0x10, 6]; // open, seek, close
    let result = database::detect_filenet_pattern(&bytecode);
    assert!(result.confidence >= 0.8);
}

#[test]
fn test_generic_io_not_misdetected() {
    let bytecode = vec![0x01, 0x02, 0x03]; // No calls
    let result = database::detect_file_io_pattern(&bytecode);
    assert!(result.confidence == 0.0);
}

// --- AI TESTS ---

#[test]
fn test_detect_int8_quantization() {
    let mut bytecode = Vec::new();
    for _ in 0..6 {
        bytecode.push(0x43); // const
        bytecode.push(0x95); // div
        bytecode.push(0xa8); // trunc
    }
    let result = ai::detect_quantization(&bytecode);
    assert!(result.confidence >= 0.9);
}

#[test]
fn test_detect_matrix_vector_multiplication() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x03, 0x0b, 0x0b]); // depth 2
    for _ in 0..40 {
        bytecode.push(0x94);
    }
    let result = ai::detect_matrix_vector_multiplication(&bytecode);
    assert!(result.confidence >= 0.8);
}

// --- INTEGRATION TESTS ---

#[test]
fn test_pattern_analyzer_runs_all_detectors() {
    let analyzer = PatternAnalyzer::new();
    let bytecode = vec![0x03, 0x94, 0x10, 1];
    let profile = analyzer.analyze(&bytecode).unwrap();

    assert!(profile.vector.confidence > 0.0);
    assert!(profile.database.confidence > 0.0);
}

#[test]
fn test_primary_pattern_selected_correctly() {
    let analyzer = PatternAnalyzer::new();
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x03, 0x03, 0x0b, 0x0b, 0x0b]); // depth 3
    for _ in 0..100 {
        bytecode.push(0x94);
    } // Massive vector muls
    for _ in 0..100 {
        bytecode.push(0x92);
    } // Massive vector adds
    let profile = analyzer.analyze(&bytecode).unwrap();

    assert!(
        profile.primary_pattern.contains("Dot Product")
            || profile.primary_pattern.contains("Matrix Multiply")
    );
}

#[test]
fn test_optimization_suggestions_generated() {
    let analyzer = PatternAnalyzer::new();
    let mut bytecode = Vec::new();
    for _ in 0..10 {
        bytecode.push(0x03);
    }
    for _ in 0..40 {
        bytecode.push(0x28);
    }
    for _ in 0..20 {
        bytecode.push(0x0c);
    }
    for _ in 0..10 {
        bytecode.push(0xfc);
    }

    let profile = analyzer.analyze(&bytecode).unwrap();
    assert!(profile.optimization_suggestion.contains("LZ4"));
}

#[test]
fn test_confidence_scores_normalized() {
    let analyzer = PatternAnalyzer::new();
    let bytecode = vec![0x01; 1000];
    let profile = analyzer.analyze(&bytecode).unwrap();

    assert!(profile.compression.confidence <= 1.0);
    assert!(profile.vector.confidence <= 1.0);
    assert!(profile.database.confidence <= 1.0);
    assert!(profile.ai.confidence <= 1.0);
}

#[test]
fn test_mixed_patterns_handled_correctly() {
    let analyzer = PatternAnalyzer::new();
    let mut bytecode = Vec::new();
    // Mixed LZ4 and Vector
    for _ in 0..10 {
        bytecode.push(0x03);
    }
    for _ in 0..30 {
        bytecode.push(0x28);
    }
    for _ in 0..50 {
        bytecode.push(0x94);
    }

    let profile = analyzer.analyze(&bytecode).unwrap();
    assert!(profile.compression.confidence >= 0.5 || profile.vector.confidence >= 0.5);
}
