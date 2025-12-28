use crate::patterns::CompressionDetectionResult;

/// Detects LZ4 decompression pattern in WASM bytecode.
///
/// ### Criteria:
/// - Loop count > 5
/// - Memory load count > 20
/// - Branch count > 10
/// - Copy operations > 5
pub fn detect_lz4_decompression(bytecode: &[u8]) -> CompressionDetectionResult {
    let loops = count_opcode(bytecode, 0x03); // loop
    let loads = count_load_opcodes(bytecode);
    let branches = count_branch_opcodes(bytecode);
    let copies = count_copy_opcodes(bytecode);

    let confidence = calculate_pattern_confidence(loops, loads, branches, copies);

    CompressionDetectionResult {
        pattern: "LZ4 Decompression".to_string(),
        confidence,
    }
}

pub fn detect_zstd_decompression(bytecode: &[u8]) -> CompressionDetectionResult {
    // Simplified stub for Zstd
    let loops = count_opcode(bytecode, 0x03);
    let loads = count_load_opcodes(bytecode);
    let confidence = if loops > 10 && loads > 50 { 0.8 } else { 0.2 };

    CompressionDetectionResult {
        pattern: "Zstd Decompression".to_string(),
        confidence,
    }
}

pub fn detect_brotli_decompression(bytecode: &[u8]) -> CompressionDetectionResult {
    // Simplified stub for Brotli
    let loops = count_opcode(bytecode, 0x03);
    let confidence = if loops > 15 { 0.7 } else { 0.1 };

    CompressionDetectionResult {
        pattern: "Brotli Decompression".to_string(),
        confidence,
    }
}

fn count_opcode(bytecode: &[u8], opcode: u8) -> usize {
    bytecode.iter().filter(|&&b| b == opcode).count()
}

fn count_load_opcodes(bytecode: &[u8]) -> usize {
    // WASM load opcodes: 0x28 to 0x35
    bytecode
        .iter()
        .filter(|&&b| (0x28..=0x35).contains(&b))
        .count()
}

fn count_branch_opcodes(bytecode: &[u8]) -> usize {
    // WASM branch opcodes: 0x0c (br), 0x0d (br_if), 0x0e (br_table)
    bytecode
        .iter()
        .filter(|&&b| (0x0c..=0x0e).contains(&b))
        .count()
}

fn count_copy_opcodes(bytecode: &[u8]) -> usize {
    // Simplified: look for memory.copy (often f32.load/store sequences in older WASM or 0xfc 0x0a)
    // For this prototype, we'll look for 0xfc as a marker for misc opcodes
    count_opcode(bytecode, 0xfc)
}

fn calculate_pattern_confidence(loops: usize, loads: usize, branches: usize, copies: usize) -> f32 {
    let score = (loops as f32 * 0.2)
        + (loads as f32 * 0.3)
        + (branches as f32 * 0.2)
        + (copies as f32 * 0.3);
    // Baseline score for LZ4 threshold (6*0.2 + 21*0.3 + 11*0.2 + 6*0.3) = 1.2 + 6.3 + 2.2 + 1.8 = 11.5
    // We want 11.5 to be approx 0.75 confidence.
    let max_score = 15.33;
    let confidence = score / max_score;
    confidence.clamp(0.0, 1.0)
}
