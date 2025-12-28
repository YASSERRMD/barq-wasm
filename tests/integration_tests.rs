use barq_wasm::analyzer::{PatternAnalyzer, PatternType}; // Use the new analyzer module
use barq_wasm::executor::BarqRuntime;
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

// --- COMPRESSION DETECTION TESTS (9) ---

#[test]
fn test_lz4_pattern_detected_in_real_code() {
    let mut bytecode = Vec::new();
    for _ in 0..3 {
        bytecode.push(0x03);
    } // Loops
    for _ in 0..25 {
        bytecode.push(0x28);
    } // Loads
    for _ in 0..15 {
        bytecode.push(0x0c);
    } // Branches
    bytecode.push(0xfc); // Copy

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);

    let lz4 = matches.iter().find(|m| m.name.contains("LZ4"));
    assert!(lz4.is_some(), "LZ4 pattern not detected");
    assert!(lz4.unwrap().confidence >= 0.75);
}

#[test]
fn test_lz4_confidence_above_threshold() {
    // Already covered partly above, but specific checks
    let mut bytecode = Vec::new();
    for _ in 0..3 {
        bytecode.push(0x03);
    }
    for _ in 0..21 {
        bytecode.push(0x28);
    }
    for _ in 0..11 {
        bytecode.push(0x0c);
    }
    // no copy, score should be >= 0.2+0.3+0.2 = 0.7 (borderline, let's add copy)
    bytecode.push(0xfc);

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches
        .iter()
        .any(|m| m.name.contains("LZ4") && m.confidence >= 0.75));
}

#[test]
fn test_zstd_pattern_detected_in_real_code() {
    let mut bytecode = Vec::new();
    for _ in 0..60 {
        bytecode.push(0x74);
    } // shifts
    for _ in 0..40 {
        bytecode.push(0x28);
    } // loads
    for _ in 0..40 {
        bytecode.push(0x71);
    } // logic

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);

    let zstd = matches.iter().find(|m| m.name.contains("Zstd"));
    assert!(zstd.is_some());
    assert!(zstd.unwrap().confidence >= 0.75);
}

#[test]
fn test_zstd_confidence_above_threshold() {
    let mut bytecode = Vec::new();
    for _ in 0..51 {
        bytecode.push(0x74);
    }
    for _ in 0..31 {
        bytecode.push(0x28);
    }
    for _ in 0..31 {
        bytecode.push(0x71);
    }

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name.contains("Zstd")));
}

#[test]
fn test_brotli_pattern_detected_in_real_code() {
    let mut bytecode = Vec::new();
    for _ in 0..20 {
        bytecode.push(0x03);
    } // loops
    for _ in 0..50 {
        bytecode.push(0x28);
    } // loads

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name.contains("Brotli")));
}

#[test]
fn test_brotli_confidence_above_threshold() {
    let mut bytecode = Vec::new();
    for _ in 0..16 {
        bytecode.push(0x03);
    }
    for _ in 0..41 {
        bytecode.push(0x28);
    }
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches
        .iter()
        .any(|m| m.name.contains("Brotli") && m.confidence >= 0.7));
}

#[test]
fn test_non_compression_not_falsely_detected() {
    let bytecode = vec![0x01, 0x02, 0x03, 0x0b]; // just a loop with nothing else
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    // Should NOT match any compression
    assert!(!matches
        .iter()
        .any(|m| m.pattern_type == PatternType::Compression));
}

#[test]
fn test_multiple_compression_patterns_ranked_correctly() {
    let mut bytecode = Vec::new();
    // Strong LZ4 signals
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

    // Weak Zstd signals
    for _ in 0..10 {
        bytecode.push(0x74);
    }

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);

    assert!(matches.len() >= 1);
    assert!(matches[0].name.contains("LZ4"));
}

#[test]
fn test_compression_optimization_hints_generated() {
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
    let lz4 = matches.iter().find(|m| m.name.contains("LZ4")).unwrap();
    assert!(!lz4.optimization_hint.is_empty());
}

// --- VECTOR DETECTION TESTS (8) ---

#[test]
fn test_dot_product_detected_in_real_code() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x0b]); // depth 1
    for _ in 0..25 {
        bytecode.push(0x94);
    }
    for _ in 0..25 {
        bytecode.push(0x92);
    }

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches
        .iter()
        .any(|m| m.name == "Dot Product" && m.confidence >= 0.8));
}

#[test]
fn test_matrix_multiply_detected_in_real_code() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x03, 0x03, 0x0b, 0x0b, 0x0b]); // depth 3
    for _ in 0..60 {
        bytecode.push(0x94);
    }
    for _ in 0..60 {
        bytecode.push(0x92);
    }

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches
        .iter()
        .any(|m| m.name == "Matrix Multiply" && m.confidence >= 0.8));
}

#[test]
fn test_cosine_similarity_detected_in_real_code() {
    let mut bytecode = Vec::new();
    for _ in 0..50 {
        bytecode.push(0x94);
    }
    for _ in 0..50 {
        bytecode.push(0x92);
    }
    bytecode.push(0x95); // div

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name == "Cosine Similarity"));
}

#[test]
fn test_vector_norm_detected_in_real_code() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x0b]);
    bytecode.push(0x91); // sqrt

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name == "Vector Norm"));
}

#[test]
fn test_matrix_transpose_detected_in_real_code() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x03, 0x0b, 0x0b]); // depth 2
    for _ in 0..30 {
        bytecode.push(0x28);
    } // loads
    for _ in 0..30 {
        bytecode.push(0x36);
    } // stores

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name == "Matrix Transpose"));
}

#[test]
fn test_vector_patterns_have_correct_confidence() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x0b]);
    for _ in 0..25 {
        bytecode.push(0x94);
    }
    for _ in 0..25 {
        bytecode.push(0x92);
    }
    // Perfect dot product profile

    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    let m = matches.iter().find(|m| m.name == "Dot Product").unwrap();
    assert!(m.confidence > 0.8);
}

#[test]
fn test_non_vector_not_falsely_detected() {
    let bytecode = vec![0x01, 0x02];
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(!matches
        .iter()
        .any(|m| m.pattern_type == PatternType::Vector));
}

#[test]
fn test_vector_optimization_hints_generated() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x0b]);
    for _ in 0..25 {
        bytecode.push(0x94);
    }
    for _ in 0..25 {
        bytecode.push(0x92);
    }
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    let m = matches.iter().find(|m| m.name == "Dot Product").unwrap();
    assert!(m.optimization_hint.contains("SIMD"));
}

// --- DATABASE DETECTION TESTS (4) ---

#[test]
fn test_mongodb_pattern_detected() {
    let bytecode = vec![0x10, 0x10, 0x10]; // calls
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name.contains("MongoDB")));
}

#[test]
fn test_filenet_pattern_detected() {
    let bytecode = vec![0x10; 10]; // many calls
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name.contains("FileNet")));
}

#[test]
fn test_database_confidence_above_threshold() {
    let bytecode = vec![0x10, 0x10, 0x10];
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    let m = matches.iter().find(|m| m.name.contains("MongoDB")).unwrap();
    assert!(m.confidence >= 0.75);
}

#[test]
fn test_non_database_not_falsely_detected() {
    let bytecode = vec![0x01, 0x02]; // no calls
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(!matches
        .iter()
        .any(|m| m.pattern_type == PatternType::Database));
}

// --- AI DETECTION TESTS (4) ---

#[test]
fn test_int8_quantization_detected() {
    let mut bytecode = Vec::new();
    for _ in 0..10 {
        bytecode.push(0x43); // const
        bytecode.push(0x95); // div
        bytecode.push(0xa8); // trunc
    }
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name == "INT8 Quantization"));
}

#[test]
fn test_convolution_layer_detected() {
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x03, 0x03, 0x0b, 0x0b, 0x0b]);
    for _ in 0..150 {
        bytecode.push(0x94);
    }
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(matches.iter().any(|m| m.name == "Convolution Layer"));
}

#[test]
fn test_ai_confidence_above_threshold() {
    let mut bytecode = Vec::new();
    for _ in 0..10 {
        bytecode.push(0x43);
        bytecode.push(0x95);
        bytecode.push(0xa8);
    }
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    let m = matches
        .iter()
        .find(|m| m.name == "INT8 Quantization")
        .unwrap();
    assert!(m.confidence >= 0.75);
}

#[test]
fn test_non_ai_not_falsely_detected() {
    let bytecode = vec![0x01, 0x02];
    let analyzer = PatternAnalyzer::default();
    let matches = analyzer.analyze(&bytecode);
    assert!(!matches.iter().any(|m| m.pattern_type == PatternType::AI));
}
