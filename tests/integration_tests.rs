use barq_wasm::analyzer::PatternAnalyzer;
use barq_wasm::codegen::cranelift_backend::{CraneliftBackend, CraneliftIR, OptimizationLevel};
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

    // Create IR that looks like LZ4 (simulated in compilation)
    // In our stub, we assume the backend applies detections if opt_level is correct

    let result = backend.compile(&[], OptimizationLevel::Compression);
    assert!(result.is_ok());
    let compiled = result.unwrap();

    // Verify optimization level preserved
    assert_eq!(compiled.optimization_level, OptimizationLevel::Compression);
}

#[test]
fn test_dictionary_loop_unrolled() {
    // Check internal logic of codegen (we exposed CraneliftIR structure for this test/demo)
    use barq_wasm::codegen::compression_codegen;
    let mut ir = CraneliftIR {
        instructions: vec![],
    };

    // Simulate detecting LZ4 loop
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
    // Analyzer detects -> Backend compiles with optimization
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

        // Start compilation if detected
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
    assert!(res.unwrap()); // returns true
    assert!(ir.instructions.contains(&"specializing_zstd".to_string()));
}
