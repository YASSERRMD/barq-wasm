use crate::analyzer::{PatternMatch, PatternType};

pub fn detect_matrix_multiply(bytecode: &[u8]) -> Option<PatternMatch> {
    let depth = count_loop_depth(bytecode);
    let muls = bytecode.iter().filter(|&&b| b == 0x94 || b == 0x95).count(); // f32.mul/div or similar
    let adds = bytecode.iter().filter(|&&b| b == 0x92 || b == 0xa0).count(); // f32.add etc

    let mut score: f32 = 0.0;
    if depth >= 3 {
        score += 0.5;
    }
    if muls > 50 {
        score += 0.25;
    }
    if adds > 50 {
        score += 0.15;
    }

    let confidence = score.min(1.0);

    if confidence >= 0.75 {
        Some(PatternMatch {
            name: "Matrix Multiply".to_string(),
            pattern_type: PatternType::Vector,
            confidence,
            optimization_hint: "Suggest tiled GEMM with 8x8 tiles".to_string(),
        })
    } else {
        None
    }
}

pub fn detect_dot_product(bytecode: &[u8]) -> Option<PatternMatch> {
    let depth = count_loop_depth(bytecode);
    let muls = bytecode.iter().filter(|&&b| b == 0x94).count();
    let adds = bytecode.iter().filter(|&&b| b == 0x92).count();

    let mut score: f32 = 0.0;
    if depth == 1 {
        score += 0.4;
    }
    if muls > 20 {
        score += 0.3;
    }
    if adds > 20 {
        score += 0.2;
    }

    let confidence = score.min(1.0);
    if confidence >= 0.8 {
        Some(PatternMatch {
            name: "Dot Product".to_string(),
            pattern_type: PatternType::Vector,
            confidence,
            optimization_hint: "Suggest SIMD with 8x parallelism (AVX2)".to_string(),
        })
    } else {
        None
    }
}

pub fn detect_vector_norm(bytecode: &[u8]) -> Option<PatternMatch> {
    let depth = count_loop_depth(bytecode);
    let has_sqrt = bytecode.contains(&0x91); // f32.sqrt

    let mut score: f32 = 0.0;
    if depth >= 1 {
        score += 0.4;
    }
    if has_sqrt {
        score += 0.5;
    }

    let confidence = score.min(1.0);

    if confidence >= 0.8 {
        Some(PatternMatch {
            name: "Vector Norm".to_string(),
            pattern_type: PatternType::Vector,
            confidence,
            optimization_hint: "Use squared-sum accumulation instructions".to_string(),
        })
    } else {
        None
    }
}

pub fn detect_cosine_similarity(bytecode: &[u8]) -> Option<PatternMatch> {
    let muls = bytecode.iter().filter(|&&b| b == 0x94).count();
    let adds = bytecode.iter().filter(|&&b| b == 0x92).count();
    let divs = bytecode.iter().filter(|&&b| b == 0x95).count();

    let mut score: f32 = 0.0;
    if muls > 40 {
        score += 0.3;
    }
    if adds > 40 {
        score += 0.3;
    }
    if divs >= 1 {
        score += 0.3;
    }

    let confidence = score.min(1.0);

    if confidence >= 0.8 {
        Some(PatternMatch {
            name: "Cosine Similarity".to_string(),
            pattern_type: PatternType::Vector,
            confidence,
            optimization_hint: "Fuse dot-product and norm calculations".to_string(),
        })
    } else {
        None
    }
}

pub fn detect_matrix_transpose(bytecode: &[u8]) -> Option<PatternMatch> {
    let depth = count_loop_depth(bytecode);
    let loads = bytecode
        .iter()
        .filter(|&&b| (0x28..=0x35).contains(&b))
        .count();
    let stores = bytecode
        .iter()
        .filter(|&&b| (0x36..=0x3e).contains(&b))
        .count();

    // Heuristic: Nested loop reading and immediatly writing
    let mut score: f32 = 0.0;
    if depth == 2 {
        score += 0.5;
    }
    if loads > 20 && stores > 20 {
        score += 0.3;
    }

    let confidence = score.min(1.0);

    if confidence >= 0.75 {
        Some(PatternMatch {
            name: "Matrix Transpose".to_string(),
            pattern_type: PatternType::Vector,
            confidence,
            optimization_hint: "Use block-tiled transpose algorithm".to_string(),
        })
    } else {
        None
    }
}

fn count_loop_depth(bytecode: &[u8]) -> usize {
    let mut max_depth = 0;
    let mut current_depth: usize = 0;
    for &b in bytecode {
        if b == 0x03 {
            current_depth += 1;
            if current_depth > max_depth {
                max_depth = current_depth;
            }
        } else if b == 0x0b {
            current_depth = current_depth.saturating_sub(1);
        }
    }
    max_depth
}
