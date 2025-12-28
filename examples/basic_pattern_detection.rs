use barq_wasm::patterns::analyzer::PatternAnalyzer;

fn main() -> anyhow::Result<()> {
    // Initialize analyzer
    let analyzer = PatternAnalyzer::new();

    // correct pattern: Matrix multiplication loop nest
    // loop (depth 1) -> loop (depth 2) -> loop (depth 3) -> end -> end -> end
    // plus float math ops
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&[0x03, 0x03, 0x03, 0x0b, 0x0b, 0x0b]); 
    for _ in 0..60 { bytecode.push(0x94); } // f32.mul
    for _ in 0..60 { bytecode.push(0x92); } // f32.add

    // Analyze
    let profile = analyzer.analyze(&bytecode)?;

    println!("Primary Pattern Detected: {}", profile.primary_pattern);
    println!("Confidence Score: {:.2}", profile.vector.confidence);
    println!("Optimization Suggestion: {}", profile.optimization_suggestion);

    Ok(())
}
