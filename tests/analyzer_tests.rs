#![cfg(all(not(target_arch = "wasm32"), feature = "analyzer"))]

//! Structural analyzer tests over a labeled WAT corpus.
//!
//! The corpus contains true positives in several compiler-output shapes,
//! structural near-misses, and unrelated code. Precision and recall are
//! measured over the whole corpus; fabricated non-WASM byte arrays must be
//! rejected, never analyzed.

use barq_wasm::analyzer::{analyze_module, PatternKind};
use barq_wasm::error::BarqError;

struct Sample {
    name: &'static str,
    wat: &'static str,
    expected: Option<PatternKind>,
}

const CONFIDENCE_THRESHOLD: f32 = 0.6;

fn corpus() -> Vec<Sample> {
    vec![
        Sample {
            name: "dot_product_indexed",
            expected: Some(PatternKind::DotProduct),
            wat: r#"(module (memory 1)
              (func (export "dot") (param $a i32) (param $b i32) (param $n i32) (result f32)
                (local $i i32) (local $sum f32)
                (block $done
                  (loop $l
                    (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
                    (local.set $sum (f32.add (local.get $sum)
                      (f32.mul
                        (f32.load (i32.add (local.get $a)
                          (i32.shl (local.get $i) (i32.const 2))))
                        (f32.load (i32.add (local.get $b)
                          (i32.shl (local.get $i) (i32.const 2)))))))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $l)))
                (local.get $sum)))"#,
        },
        Sample {
            name: "dot_product_pointer_bump",
            expected: Some(PatternKind::DotProduct),
            wat: r#"(module (memory 1)
              (func (export "dot") (param $a i32) (param $b i32) (param $count i32) (result f32)
                (local $sum f32)
                (block $done
                  (loop $l
                    (br_if $done (i32.eqz (local.get $count)))
                    (local.set $sum (f32.add
                      (f32.mul (f32.load (local.get $a)) (f32.load (local.get $b)))
                      (local.get $sum)))
                    (local.set $a (i32.add (local.get $a) (i32.const 4)))
                    (local.set $b (i32.add (local.get $b) (i32.const 4)))
                    (local.set $count (i32.add (local.get $count) (i32.const -1)))
                    (br $l)))
                (local.get $sum)))"#,
        },
        Sample {
            name: "dot_product_with_unrelated_arithmetic",
            expected: Some(PatternKind::DotProduct),
            wat: r#"(module (memory 1)
              (func (export "dot_noisy") (param $a i32) (param $b i32) (param $n i32) (result f32)
                (local $i i32) (local $sum f32) (local $junk i32)
                (block $done
                  (loop $l
                    (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
                    (local.set $junk (i32.mul (local.get $i) (i32.const 17)))
                    (local.set $junk (i32.add (local.get $junk) (i32.const 3)))
                    (local.set $sum (f32.add (local.get $sum)
                      (f32.mul
                        (f32.load (i32.add (local.get $a)
                          (i32.shl (local.get $i) (i32.const 2))))
                        (f32.load (i32.add (local.get $b)
                          (i32.shl (local.get $i) (i32.const 2)))))))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $l)))
                (local.get $sum)))"#,
        },
        Sample {
            // Same pointer on both sides: sum of squares, not a dot product.
            name: "near_miss_sum_of_squares",
            expected: None,
            wat: r#"(module (memory 1)
              (func (export "sumsq") (param $a i32) (param $n i32) (result f32)
                (local $i i32) (local $sum f32)
                (block $done
                  (loop $l
                    (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
                    (local.set $sum (f32.add (local.get $sum)
                      (f32.mul
                        (f32.load (i32.add (local.get $a)
                          (i32.shl (local.get $i) (i32.const 2))))
                        (f32.load (i32.add (local.get $a)
                          (i32.shl (local.get $i) (i32.const 2)))))))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $l)))
                (local.get $sum)))"#,
        },
        Sample {
            // Elementwise product stored to memory: no reduction.
            name: "near_miss_elementwise_multiply",
            expected: None,
            wat: r#"(module (memory 1)
              (func (export "mul") (param $a i32) (param $b i32) (param $c i32) (param $n i32)
                (local $i i32)
                (block $done
                  (loop $l
                    (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
                    (f32.store (i32.add (local.get $c)
                        (i32.shl (local.get $i) (i32.const 2)))
                      (f32.mul
                        (f32.load (i32.add (local.get $a)
                          (i32.shl (local.get $i) (i32.const 2))))
                        (f32.load (i32.add (local.get $b)
                          (i32.shl (local.get $i) (i32.const 2))))))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $l)))))"#,
        },
        Sample {
            // Plain sum: only one load stream, no multiply of loads.
            name: "near_miss_plain_sum",
            expected: None,
            wat: r#"(module (memory 1)
              (func (export "sum") (param $a i32) (param $n i32) (result f32)
                (local $i i32) (local $sum f32)
                (block $done
                  (loop $l
                    (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
                    (local.set $sum (f32.add (local.get $sum)
                      (f32.load (i32.add (local.get $a)
                        (i32.shl (local.get $i) (i32.const 2))))))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $l)))
                (local.get $sum)))"#,
        },
        Sample {
            name: "matrix_multiply_three_loops",
            expected: Some(PatternKind::MatrixMultiply),
            wat: r#"(module (memory 1)
              (func (export "matmul") (param $a i32) (param $b i32) (param $c i32) (param $n i32)
                (local $i i32) (local $j i32) (local $k i32) (local $sum f32)
                (block $done_i (loop $li
                  (br_if $done_i (i32.ge_s (local.get $i) (local.get $n)))
                  (local.set $j (i32.const 0))
                  (block $done_j (loop $lj
                    (br_if $done_j (i32.ge_s (local.get $j) (local.get $n)))
                    (local.set $sum (f32.const 0))
                    (local.set $k (i32.const 0))
                    (block $done_k (loop $lk
                      (br_if $done_k (i32.ge_s (local.get $k) (local.get $n)))
                      (local.set $sum (f32.add (local.get $sum)
                        (f32.mul
                          (f32.load (i32.add (local.get $a) (i32.shl
                            (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $k))
                            (i32.const 2))))
                          (f32.load (i32.add (local.get $b) (i32.shl
                            (i32.add (i32.mul (local.get $k) (local.get $n)) (local.get $j))
                            (i32.const 2)))))))
                      (local.set $k (i32.add (local.get $k) (i32.const 1)))
                      (br $lk)))
                    (f32.store (i32.add (local.get $c) (i32.shl
                      (i32.add (i32.mul (local.get $i) (local.get $n)) (local.get $j))
                      (i32.const 2)))
                      (local.get $sum))
                    (local.set $j (i32.add (local.get $j) (i32.const 1)))
                    (br $lj)))
                  (local.set $i (i32.add (local.get $i) (i32.const 1)))
                  (br $li)))))"#,
        },
        Sample {
            name: "quantization_scale_clamp_store8",
            expected: Some(PatternKind::Quantization),
            wat: r#"(module (memory 1)
              (func (export "quant") (param $src i32) (param $dst i32) (param $n i32) (param $inv f32)
                (local $i i32)
                (block $done
                  (loop $l
                    (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
                    (i32.store8 (i32.add (local.get $dst) (local.get $i))
                      (i32.trunc_sat_f32_s
                        (f32.max (f32.const -128)
                          (f32.min (f32.const 127)
                            (f32.mul
                              (f32.load (i32.add (local.get $src)
                                (i32.shl (local.get $i) (i32.const 2))))
                              (local.get $inv))))))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $l)))))"#,
        },
        Sample {
            // Byte copy loop: narrow store but no float involvement.
            name: "near_miss_byte_copy",
            expected: None,
            wat: r#"(module (memory 1)
              (func (export "copy") (param $src i32) (param $dst i32) (param $n i32)
                (local $i i32)
                (block $done
                  (loop $l
                    (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
                    (i32.store8 (i32.add (local.get $dst) (local.get $i))
                      (i32.load8_u (i32.add (local.get $src) (local.get $i))))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $l)))))"#,
        },
        Sample {
            name: "unrelated_no_loops",
            expected: None,
            wat: r#"(module
              (func (export "poly") (param $x f32) (result f32)
                (f32.add (f32.mul (local.get $x) (local.get $x)) (f32.const 1))))"#,
        },
    ]
}

fn top_pattern(wat: &str) -> Option<PatternKind> {
    let bytes = wat::parse_str(wat).expect("corpus WAT must assemble");
    let analysis = analyze_module(&bytes).expect("analysis must succeed on valid modules");
    analysis
        .candidates()
        .filter(|(_, c)| c.confidence >= CONFIDENCE_THRESHOLD)
        .max_by(|a, b| a.1.confidence.total_cmp(&b.1.confidence))
        .map(|(_, c)| c.pattern)
}

#[test]
fn corpus_precision_and_recall() {
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut false_negatives = 0usize;
    let mut failures = vec![];

    for sample in corpus() {
        let got = top_pattern(sample.wat);
        match (sample.expected, got) {
            (Some(want), Some(g)) if g == want => true_positives += 1,
            (None, None) => {}
            (want, got) => {
                if got.is_some() {
                    false_positives += 1;
                } else {
                    false_negatives += 1;
                }
                failures.push(format!("{}: expected {want:?}, got {got:?}", sample.name));
            }
        }
    }

    let precision = true_positives as f32 / (true_positives + false_positives).max(1) as f32;
    let recall = true_positives as f32 / (true_positives + false_negatives).max(1) as f32;
    println!("precision={precision} recall={recall}");
    assert!(
        failures.is_empty(),
        "precision={precision} recall={recall}; failures:\n{}",
        failures.join("\n")
    );
    assert_eq!(precision, 1.0, "no false positives allowed in the corpus");
    assert_eq!(recall, 1.0, "all curated true positives must be detected");
}

#[test]
fn candidates_carry_structural_evidence() {
    let bytes = wat::parse_str(corpus()[0].wat).unwrap();
    let analysis = analyze_module(&bytes).unwrap();
    let (_, candidate) = analysis
        .candidates()
        .next()
        .expect("dot product candidate expected");
    assert!(candidate.confidence > 0.0 && candidate.confidence <= 1.0);
    assert!(
        candidate.evidence.len() >= 3,
        "evidence must list the structural features found"
    );
    let features: Vec<_> = candidate.evidence.iter().map(|e| e.feature).collect();
    assert!(features.contains(&"multiply_accumulate_reduction"));
    assert!(features.contains(&"two_independent_linear_loads"));
}

#[test]
fn confidence_reflects_evidence_count_not_a_constant() {
    // The noisy variant and the clean variant must both be detected, and
    // confidence must equal satisfied/total requirements — not a fixed 0.8.
    for sample in corpus().iter().take(3) {
        let bytes = wat::parse_str(sample.wat).unwrap();
        let analysis = analyze_module(&bytes).unwrap();
        for (_, c) in analysis.candidates() {
            let reconstructed = c.evidence.len() as f32 / 5.0;
            assert!(
                (c.confidence - reconstructed).abs() < 1e-6,
                "confidence {} must equal evidence/total {}",
                c.confidence,
                reconstructed
            );
        }
    }
}

#[test]
fn rejects_fabricated_byte_arrays() {
    // The old analyzer accepted arbitrary opcode-count arrays. The real one
    // must refuse anything that is not a parseable module.
    let fabricated = vec![0x8Cu8; 64];
    assert!(matches!(
        analyze_module(&fabricated),
        Err(BarqError::Validation(_))
    ));
    let empty: &[u8] = &[];
    assert!(analyze_module(empty).is_err());
}

#[test]
fn analyzes_module_metadata() {
    let bytes = wat::parse_str(
        r#"(module
          (import "env" "log" (func (param i32)))
          (memory (export "memory") 1)
          (func (export "f") (result i32) i32.const 1))"#,
    )
    .unwrap();
    let analysis = analyze_module(&bytes).unwrap();
    assert_eq!(analysis.imports.len(), 1);
    assert!(analysis.exports.iter().any(|e| e.name == "f"));
    // Imported function shifts the defined function's index.
    assert_eq!(analysis.functions[0].function_index, 1);
    assert_eq!(analysis.functions[0].export_name.as_deref(), Some("f"));
}
