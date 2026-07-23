//! Structural pattern detection over symbolic loop facts.
//!
//! Every candidate carries the concrete evidence that produced it, and
//! confidence is computed from satisfied structural requirements — never a
//! hard-coded number. Core requirements are mandatory: without them no
//! candidate is emitted at all.

use super::engine::{BodyFacts, LoopFacts};
use super::expr::{BinOp, ConvertOp, Expr, LoadKind};
use std::collections::BTreeSet;
use std::rc::Rc;

/// Recognized computational patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    DotProduct,
    MatrixMultiply,
    Quantization,
}

impl std::fmt::Display for PatternKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternKind::DotProduct => write!(f, "dot_product"),
            PatternKind::MatrixMultiply => write!(f, "matrix_multiply"),
            PatternKind::Quantization => write!(f, "quantization"),
        }
    }
}

/// One piece of structural evidence supporting a candidate.
#[derive(Debug, Clone)]
pub struct PatternEvidence {
    pub feature: &'static str,
    pub detail: String,
}

/// A detected pattern candidate.
#[derive(Debug, Clone)]
pub struct PatternCandidate {
    pub pattern: PatternKind,
    /// satisfied structural requirements / total requirements. Never fixed.
    pub confidence: f32,
    pub evidence: Vec<PatternEvidence>,
    /// Index into the function's loop list this candidate anchors to.
    pub loop_index: Option<usize>,
}

/// A multiply of two loads found inside an accumulation or store.
struct MulOfLoads {
    bases_a: BTreeSet<u32>,
    bases_b: BTreeSet<u32>,
}

/// Find `x * y` where both operands are (or contain) float loads, and report
/// the base locals of each side's address expression.
fn find_mul_of_loads(expr: &Expr) -> Option<MulOfLoads> {
    let mut found = None;
    expr.walk(&mut |e| {
        if found.is_some() {
            return;
        }
        if let Expr::Bin {
            op: BinOp::FMul,
            lhs,
            rhs,
        } = e
        {
            let lhs_loads = lhs.loads();
            let rhs_loads = rhs.loads();
            let float = |k: &LoadKind| matches!(k, LoadKind::F32 | LoadKind::F64);
            if lhs_loads.iter().any(|(_, k)| float(k)) && rhs_loads.iter().any(|(_, k)| float(k)) {
                let bases = |loads: &[(Rc<Expr>, LoadKind)]| {
                    loads
                        .iter()
                        .filter(|(_, k)| float(k))
                        .flat_map(|(a, _)| a.locals())
                        .collect::<BTreeSet<u32>>()
                };
                found = Some(MulOfLoads {
                    bases_a: bases(&lhs_loads),
                    bases_b: bases(&rhs_loads),
                });
            }
        }
    });
    found
}

/// A local that accumulates: `local.set X (f32.add (.. local.get X ..) ..)`.
fn accumulator_with_mul(lp: &LoopFacts) -> Option<(u32, MulOfLoads)> {
    for set in &lp.sets {
        if let Expr::Bin {
            op: BinOp::FAdd,
            lhs,
            rhs,
        } = &*set.value
        {
            let reads_self =
                lhs.locals().contains(&set.local) || rhs.locals().contains(&set.local);
            if !reads_self {
                continue;
            }
            if let Some(mul) = find_mul_of_loads(&set.value) {
                return Some((set.local, mul));
            }
        }
    }
    None
}

/// Induction variable: `local.set X (i32.add (local.get X) (const))`.
fn induction_locals(lp: &LoopFacts) -> Vec<u32> {
    let mut out = vec![];
    for set in &lp.sets {
        if let Expr::Bin {
            op: BinOp::IAdd,
            lhs,
            rhs,
        } = &*set.value
        {
            let self_plus_const = (matches!(&**lhs, Expr::Local(i) if *i == set.local)
                && matches!(&**rhs, Expr::Const(_)))
                || (matches!(&**rhs, Expr::Local(i) if *i == set.local)
                    && matches!(&**lhs, Expr::Const(_)));
            if self_plus_const {
                out.push(set.local);
            }
        }
    }
    out
}

/// Dot-product detection on one innermost loop.
pub fn detect_dot_product(lp: &LoopFacts, loop_index: usize) -> Option<PatternCandidate> {
    const TOTAL: f32 = 5.0;
    let mut evidence = vec![];

    // Core: multiply-accumulate into a local, over two float loads.
    let (acc_local, mul) = accumulator_with_mul(lp)?;
    evidence.push(PatternEvidence {
        feature: "multiply_accumulate_reduction",
        detail: format!("local {acc_local} accumulates a product of float loads"),
    });

    // Core: the two multiplied loads use different base locals.
    if mul.bases_a.is_empty() || mul.bases_b.is_empty() || mul.bases_a == mul.bases_b {
        return None;
    }
    if mul.bases_a.intersection(&mul.bases_b).next().is_some()
        && mul.bases_a.union(&mul.bases_b).count() <= 2
    {
        // Shared single pointer on both sides (e.g. sum of squares) is not a
        // two-vector dot product; require at least one distinct base.
        return None;
    }
    evidence.push(PatternEvidence {
        feature: "two_independent_linear_loads",
        detail: format!(
            "load bases {:?} vs {:?}",
            mul.bases_a, mul.bases_b
        ),
    });

    if lp.innermost {
        evidence.push(PatternEvidence {
            feature: "single_primary_loop",
            detail: "innermost loop with no nested loops".to_string(),
        });
    }
    let ind = induction_locals(lp);
    if !ind.is_empty() && !lp.branch_conditions.is_empty() {
        evidence.push(PatternEvidence {
            feature: "induction_variable",
            detail: format!("locals {ind:?} advance by constant with loop branch"),
        });
    }
    let float_store = lp
        .stores
        .iter()
        .any(|s| matches!(s.kind, LoadKind::F32 | LoadKind::F64));
    if !float_store {
        evidence.push(PatternEvidence {
            feature: "no_output_store",
            detail: "reduction loop writes no float memory".to_string(),
        });
    }

    Some(PatternCandidate {
        pattern: PatternKind::DotProduct,
        confidence: evidence.len() as f32 / TOTAL,
        evidence,
        loop_index: Some(loop_index),
    })
}

/// Matrix-multiply detection over a whole function body.
pub fn detect_matrix_multiply(body: &BodyFacts) -> Option<PatternCandidate> {
    const TOTAL: f32 = 5.0;
    let mut evidence = vec![];

    // Core: three nested loops.
    if body.max_loop_depth < 3 {
        return None;
    }
    evidence.push(PatternEvidence {
        feature: "three_nested_loops",
        detail: format!("max loop depth {}", body.max_loop_depth),
    });

    // Core: an innermost loop with a multiply-accumulate over two regions.
    let inner = body
        .loops
        .iter()
        .enumerate()
        .filter(|(_, l)| l.innermost && l.loop_depth >= 3)
        .find_map(|(i, l)| {
            accumulator_with_mul(l)
                .map(|(local, mul)| (i, local, mul))
                .or_else(|| store_accumulate(l).map(|mul| (i, u32::MAX, mul)))
        });
    let (loop_index, _acc, mul) = inner?;
    evidence.push(PatternEvidence {
        feature: "multiply_accumulate",
        detail: "innermost loop accumulates a product of float loads".to_string(),
    });

    // Core: two distinct input regions.
    if mul.bases_a.is_empty() || mul.bases_b.is_empty() || mul.bases_a == mul.bases_b {
        return None;
    }
    evidence.push(PatternEvidence {
        feature: "two_input_regions",
        detail: format!("bases {:?} vs {:?}", mul.bases_a, mul.bases_b),
    });

    // Strided addressing: some load/store address uses a multiply or shift.
    let strided = body.loops.iter().any(|l| {
        l.loads
            .iter()
            .map(|ld| ld.addr.clone())
            .chain(l.stores.iter().map(|s| s.addr.clone()))
            .any(|addr| {
                let mut found = false;
                addr.walk(&mut |e| {
                    if matches!(
                        e,
                        Expr::Bin {
                            op: BinOp::IMul | BinOp::IShl,
                            ..
                        }
                    ) {
                        found = true;
                    }
                });
                found
            })
    });
    if strided {
        evidence.push(PatternEvidence {
            feature: "strided_addressing",
            detail: "address computation uses multiply/shift strides".to_string(),
        });
    }

    // Output store somewhere in the body.
    if body
        .all_stores
        .iter()
        .any(|s| matches!(s.kind, LoadKind::F32 | LoadKind::F64))
    {
        evidence.push(PatternEvidence {
            feature: "output_store",
            detail: "float result written to memory".to_string(),
        });
    }

    Some(PatternCandidate {
        pattern: PatternKind::MatrixMultiply,
        confidence: evidence.len() as f32 / TOTAL,
        evidence,
        loop_index: Some(loop_index),
    })
}

/// `c[..] = load(c_addr) + a*b` style accumulation directly through memory.
fn store_accumulate(lp: &LoopFacts) -> Option<MulOfLoads> {
    for store in &lp.stores {
        if !matches!(store.kind, LoadKind::F32 | LoadKind::F64) {
            continue;
        }
        if let Expr::Bin {
            op: BinOp::FAdd, ..
        } = &*store.value
        {
            if store.value.loads().len() >= 3 {
                if let Some(mul) = find_mul_of_loads(&store.value) {
                    return Some(mul);
                }
            }
        }
    }
    None
}

/// Quantization detection on one loop.
pub fn detect_quantization(lp: &LoopFacts, loop_index: usize) -> Option<PatternCandidate> {
    const TOTAL: f32 = 5.0;
    let mut evidence = vec![];

    // Core: a float load feeds the loop.
    let float_load = lp
        .loads
        .iter()
        .any(|l| matches!(l.kind, LoadKind::F32 | LoadKind::F64));
    if !float_load {
        return None;
    }
    evidence.push(PatternEvidence {
        feature: "float_load",
        detail: "loop reads f32/f64 memory".to_string(),
    });

    // Core: narrow integer store of a converted value.
    let narrow_store = lp.stores.iter().find(|s| {
        matches!(s.kind, LoadKind::I8 | LoadKind::I16) && {
            let mut has_convert = false;
            s.value.walk(&mut |e| {
                if matches!(
                    e,
                    Expr::Convert {
                        op: ConvertOp::F32ToI32,
                        ..
                    }
                ) {
                    has_convert = true;
                }
            });
            has_convert
        }
    });
    let narrow_store = narrow_store?;
    evidence.push(PatternEvidence {
        feature: "narrow_integer_store",
        detail: "i8/i16 store of a float->int converted value".to_string(),
    });
    evidence.push(PatternEvidence {
        feature: "numeric_conversion",
        detail: "float to integer truncation in stored value".to_string(),
    });

    // Scale arithmetic: multiply or divide on the float path of the store.
    let mut has_scale = false;
    narrow_store.value.walk(&mut |e| {
        if matches!(
            e,
            Expr::Bin {
                op: BinOp::FMul | BinOp::FDiv,
                ..
            }
        ) {
            has_scale = true;
        }
    });
    if has_scale {
        evidence.push(PatternEvidence {
            feature: "scale_arithmetic",
            detail: "float multiply/divide before conversion".to_string(),
        });
    }

    // Saturation/clamping: select or float min/max in the stored value.
    let mut has_clamp = false;
    narrow_store.value.walk(&mut |e| {
        if matches!(
            e,
            Expr::Select { .. }
                | Expr::Bin {
                    op: BinOp::FMin | BinOp::FMax,
                    ..
                }
        ) {
            has_clamp = true;
        }
    });
    if has_clamp {
        evidence.push(PatternEvidence {
            feature: "saturation_clamp",
            detail: "select/min/max clamping before the narrow store".to_string(),
        });
    }

    Some(PatternCandidate {
        pattern: PatternKind::Quantization,
        confidence: evidence.len() as f32 / TOTAL,
        evidence,
        loop_index: Some(loop_index),
    })
}
