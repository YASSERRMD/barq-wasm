use crate::patterns::AIDetectionResult;

pub fn detect_quantization(bytecode: &[u8]) -> AIDetectionResult {
    let (consts, divs, truncs) = count_quantization_operations(bytecode);

    // Pattern: f32.const -> f32.div -> i32.trunc
    let mut confidence = 0.0;
    if consts > 0 {
        confidence += 0.2;
    }
    if divs > 0 {
        confidence += 0.3;
    }
    if truncs > 0 {
        confidence += 0.4;
    }

    // Check for sequence-like density
    if consts > 5 && divs > 5 && truncs > 5 {
        confidence = 0.95;
    }

    AIDetectionResult {
        pattern: "Int8 Quantization".to_string(),
        confidence,
    }
}

pub fn detect_matrix_vector_multiplication(bytecode: &[u8]) -> AIDetectionResult {
    let depth = detect_loop_nesting_depth(bytecode);
    let muls = bytecode.iter().filter(|&&b| b == 0x94).count(); // f32.mul

    let confidence = if depth == 2 && muls > 30 { 0.85 } else { 0.1 };

    AIDetectionResult {
        pattern: "Matrix-Vector Multiply".to_string(),
        confidence,
    }
}

pub fn detect_convolution(bytecode: &[u8]) -> AIDetectionResult {
    let depth = detect_loop_nesting_depth(bytecode);
    let muls = bytecode.iter().filter(|&&b| b == 0x94).count();

    let confidence = if depth >= 3 && muls > 100 { 0.9 } else { 0.2 };

    AIDetectionResult {
        pattern: "2D/3D Convolution".to_string(),
        confidence,
    }
}

pub fn detect_softmax(bytecode: &[u8]) -> AIDetectionResult {
    let divs = bytecode.iter().filter(|&&b| b == 0x95).count();
    let has_exp = bytecode.contains(&0x10); // Assume exp is a call

    let confidence = if has_exp && divs > 0 { 0.75 } else { 0.1 };

    AIDetectionResult {
        pattern: "Softmax Activation".to_string(),
        confidence,
    }
}

fn count_quantization_operations(bytecode: &[u8]) -> (usize, usize, usize) {
    let consts = bytecode.iter().filter(|&&b| b == 0x43).count();
    let divs = bytecode.iter().filter(|&&b| b == 0x95).count();
    let truncs = bytecode.iter().filter(|&&b| b == 0xa8).count();

    (consts, divs, truncs)
}

fn detect_loop_nesting_depth(bytecode: &[u8]) -> usize {
    let mut max_depth = 0;
    let mut current_depth: usize = 0;

    for &byte in bytecode {
        if byte == 0x03 {
            // loop
            current_depth += 1;
            if current_depth > max_depth {
                max_depth = current_depth;
            }
        } else if byte == 0x0b {
            // end
            current_depth = current_depth.saturating_sub(1);
        }
    }

    max_depth
}
