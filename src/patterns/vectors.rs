use crate::patterns::VectorDetectionResult;

pub fn detect_matrix_multiply(bytecode: &[u8]) -> VectorDetectionResult {
    let depth = detect_loop_nesting_depth(bytecode);
    let (adds, muls, _) = count_float_operations(bytecode);

    let mut confidence = 0.0;
    if depth >= 3 {
        confidence += 0.4;
    }
    if muls > 50 {
        confidence += 0.3;
    }
    if adds > 50 {
        confidence += 0.3;
    }

    VectorDetectionResult {
        pattern: "Matrix Multiply".to_string(),
        confidence,
    }
}

pub fn detect_dot_product(bytecode: &[u8]) -> VectorDetectionResult {
    let depth = detect_loop_nesting_depth(bytecode);
    let (adds, muls, _) = count_float_operations(bytecode);

    let mut confidence = 0.0;
    if depth == 1 {
        confidence += 0.4;
    }
    if muls > 20 {
        confidence += 0.3;
    }
    if adds > 20 {
        confidence += 0.3;
    }

    VectorDetectionResult {
        pattern: "Dot Product".to_string(),
        confidence,
    }
}

pub fn detect_vector_norm(bytecode: &[u8]) -> VectorDetectionResult {
    let (_, _, has_sqrt) = count_float_operations(bytecode);
    let depth = detect_loop_nesting_depth(bytecode);

    let confidence = if depth >= 1 && has_sqrt > 0 { 0.8 } else { 0.1 };

    VectorDetectionResult {
        pattern: "Vector Norm".to_string(),
        confidence,
    }
}

pub fn detect_cosine_similarity(bytecode: &[u8]) -> VectorDetectionResult {
    let (adds, muls, _) = count_float_operations(bytecode);
    let divs = bytecode.iter().filter(|&&b| b == 0x95).count(); // f32.div

    let confidence = if muls > 40 && adds > 40 && divs >= 1 {
        0.9
    } else {
        0.2
    };

    VectorDetectionResult {
        pattern: "Cosine Similarity".to_string(),
        confidence,
    }
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

fn count_float_operations(bytecode: &[u8]) -> (usize, usize, usize) {
    let adds = bytecode.iter().filter(|&&b| b == 0x92).count();
    let muls = bytecode.iter().filter(|&&b| b == 0x94).count();
    let sqrts = bytecode.iter().filter(|&&b| b == 0x91).count();

    (adds, muls, sqrts)
}
