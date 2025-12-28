use crate::analyzer::{PatternMatch, PatternType};

pub fn detect_int8_quantization(bytecode: &[u8]) -> Option<PatternMatch> {
    // Quantization:
    // f32.const (scale), f32.div, i32.trunc
    let consts = bytecode.iter().filter(|&&b| b == 0x43).count();
    let divs = bytecode.iter().filter(|&&b| b == 0x95).count();
    let truncs = bytecode.iter().filter(|&&b| b == 0xa8).count();

    let mut score: f32 = 0.0;
    if consts > 5 {
        score += 0.2;
    }
    if divs > 5 {
        score += 0.3;
    }
    if truncs > 5 {
        score += 0.4;
    }

    let confidence = score.min(1.0);

    if confidence >= 0.75 {
        Some(PatternMatch {
            name: "INT8 Quantization".to_string(),
            pattern_type: PatternType::AI,
            confidence,
            optimization_hint: "Use INT8 tensor instructions (VNNI)".to_string(),
        })
    } else {
        None
    }
}

pub fn detect_convolution_layer(bytecode: &[u8]) -> Option<PatternMatch> {
    // Convolution:
    // Triple nested loops (width, height, channels) -> depth 3
    // Inner loop mul/add

    let depth = count_loop_depth(bytecode);
    let muls = bytecode.iter().filter(|&&b| b == 0x94).count();

    let mut score: f32 = 0.0;
    if depth >= 3 {
        score += 0.5;
    }
    if muls > 100 {
        score += 0.3;
    }

    let confidence = score.min(1.0);

    if confidence >= 0.7 {
        Some(PatternMatch {
            name: "Convolution Layer".to_string(),
            pattern_type: PatternType::AI,
            confidence,
            optimization_hint: "Use im2col transformation and GEMM".to_string(),
        })
    } else {
        None
    }
}

pub fn detect_attention_layer(bytecode: &[u8]) -> Option<PatternMatch> {
    // Attention:
    // Matrix mul (QK^T) -> Softmax -> Matrix mul (V)
    // Heuristic: Matrix Mul pattern + Softmax pattern (exp + div)

    let muls = bytecode.iter().filter(|&&b| b == 0x94).count();
    let has_exp = bytecode.contains(&0x10); // simulating call to exp
    let divs = bytecode.iter().filter(|&&b| b == 0x95).count();

    let mut score: f32 = 0.0;
    if muls > 50 {
        score += 0.3;
    }
    if has_exp && divs > 0 {
        score += 0.5;
    }

    let confidence = score.min(1.0);

    if confidence >= 0.7 {
        Some(PatternMatch {
            name: "Attention Layer".to_string(),
            pattern_type: PatternType::AI,
            confidence,
            optimization_hint: "Fuse Softmax and MatMul kernels".to_string(),
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
