//! Symbolic execution of function bodies into structural facts.
//!
//! Walks real parsed operators (wasmparser) and reconstructs, per loop:
//! which addresses are loaded, what values are stored where, which locals
//! accumulate, and which locals act as induction variables. Unmodeled
//! operators degrade to `Expr::Unknown` — they never abort analysis and
//! never fabricate structure.

use super::expr::{BinOp, ConvertOp, Expr, LoadKind};
use std::rc::Rc;
use wasmparser::Operator;

/// A load event observed inside a loop.
#[derive(Debug, Clone)]
pub struct LoadEvent {
    pub addr: Rc<Expr>,
    pub kind: LoadKind,
}

/// A store event observed inside a loop.
#[derive(Debug, Clone)]
pub struct StoreEvent {
    pub addr: Rc<Expr>,
    pub value: Rc<Expr>,
    pub kind: LoadKind,
}

/// A `local.set`/`local.tee` event.
#[derive(Debug, Clone)]
pub struct SetEvent {
    pub local: u32,
    pub value: Rc<Expr>,
}

/// Facts gathered for one `loop` construct.
#[derive(Debug, Clone, Default)]
pub struct LoopFacts {
    /// 1 = outermost loop in the function, increasing with loop nesting.
    pub loop_depth: usize,
    /// True when no other loop is nested inside this one.
    pub innermost: bool,
    pub loads: Vec<LoadEvent>,
    pub stores: Vec<StoreEvent>,
    pub sets: Vec<SetEvent>,
    pub branch_conditions: Vec<Rc<Expr>>,
    pub operator_count: usize,
}

/// Facts for a whole function body.
#[derive(Debug, Clone, Default)]
pub struct BodyFacts {
    pub loops: Vec<LoopFacts>,
    pub max_loop_depth: usize,
    pub operator_count: usize,
    /// Stores occurring anywhere in the body (incl. outside loops).
    pub all_stores: Vec<StoreEvent>,
}

pub fn analyze_body(operators: &[Operator]) -> BodyFacts {
    let mut facts = BodyFacts::default();
    let mut stack: Vec<Rc<Expr>> = Vec::new();
    // Indices into facts.loops for loops currently open, innermost last.
    let mut open_loops: Vec<usize> = Vec::new();
    // Control nesting for matching `end`s: true = loop, false = block/if.
    let mut control: Vec<bool> = Vec::new();

    let unknown = || Rc::new(Expr::Unknown);
    let pop = |stack: &mut Vec<Rc<Expr>>| stack.pop().unwrap_or_else(|| Rc::new(Expr::Unknown));

    for op in operators {
        facts.operator_count += 1;
        for &li in &open_loops {
            facts.loops[li].operator_count += 1;
        }
        match op {
            Operator::Loop { .. } => {
                let depth = open_loops.len() + 1;
                facts.loops.push(LoopFacts {
                    loop_depth: depth,
                    innermost: true,
                    ..LoopFacts::default()
                });
                let idx = facts.loops.len() - 1;
                // Any enclosing loop is no longer innermost.
                for &outer in &open_loops {
                    facts.loops[outer].innermost = false;
                }
                open_loops.push(idx);
                facts.max_loop_depth = facts.max_loop_depth.max(depth);
                control.push(true);
            }
            Operator::Block { .. } | Operator::If { .. } => {
                if matches!(op, Operator::If { .. }) {
                    let _cond = pop(&mut stack);
                }
                control.push(false);
            }
            Operator::Else => {}
            Operator::End => {
                if let Some(was_loop) = control.pop() {
                    if was_loop {
                        open_loops.pop();
                    }
                }
            }
            Operator::Br { .. } => {}
            Operator::BrIf { .. } | Operator::BrTable { .. } => {
                let cond = pop(&mut stack);
                if let Some(&li) = open_loops.last() {
                    facts.loops[li].branch_conditions.push(cond);
                }
            }
            Operator::Return => {}
            Operator::Call { .. } | Operator::CallIndirect { .. } => {
                // Unknown arity without type info: conservatively clear the
                // stack and push one unknown result.
                stack.clear();
                stack.push(unknown());
            }
            Operator::Drop => {
                let _ = pop(&mut stack);
            }
            Operator::Select => {
                let b = pop(&mut stack);
                let a = pop(&mut stack);
                let _cond = pop(&mut stack);
                stack.push(Rc::new(Expr::Select { a, b }));
            }
            Operator::LocalGet { local_index } => {
                stack.push(Rc::new(Expr::Local(*local_index)));
            }
            Operator::LocalSet { local_index } | Operator::LocalTee { local_index } => {
                let value = pop(&mut stack);
                let event = SetEvent {
                    local: *local_index,
                    value: value.clone(),
                };
                if let Some(&li) = open_loops.last() {
                    facts.loops[li].sets.push(event);
                }
                if matches!(op, Operator::LocalTee { .. }) {
                    stack.push(value);
                }
            }
            Operator::I32Const { value } => stack.push(Rc::new(Expr::Const(*value as f64))),
            Operator::I64Const { value } => stack.push(Rc::new(Expr::Const(*value as f64))),
            Operator::F32Const { value } => {
                stack.push(Rc::new(Expr::Const(f32::from_bits(value.bits()) as f64)))
            }
            Operator::F64Const { value } => {
                stack.push(Rc::new(Expr::Const(f64::from_bits(value.bits()))))
            }
            Operator::F32Load { .. }
            | Operator::F64Load { .. }
            | Operator::I32Load { .. }
            | Operator::I64Load { .. }
            | Operator::I32Load8S { .. }
            | Operator::I32Load8U { .. }
            | Operator::I32Load16S { .. }
            | Operator::I32Load16U { .. } => {
                let addr = pop(&mut stack);
                let kind = match op {
                    Operator::F32Load { .. } => LoadKind::F32,
                    Operator::F64Load { .. } => LoadKind::F64,
                    Operator::I32Load { .. } => LoadKind::I32,
                    Operator::I64Load { .. } => LoadKind::I64,
                    Operator::I32Load8S { .. } | Operator::I32Load8U { .. } => LoadKind::I8,
                    _ => LoadKind::I16,
                };
                let expr = Rc::new(Expr::Load {
                    addr: addr.clone(),
                    kind,
                });
                if let Some(&li) = open_loops.last() {
                    facts.loops[li].loads.push(LoadEvent { addr, kind });
                }
                stack.push(expr);
            }
            Operator::F32Store { .. }
            | Operator::F64Store { .. }
            | Operator::I32Store { .. }
            | Operator::I64Store { .. }
            | Operator::I32Store8 { .. }
            | Operator::I32Store16 { .. } => {
                let value = pop(&mut stack);
                let addr = pop(&mut stack);
                let kind = match op {
                    Operator::F32Store { .. } => LoadKind::F32,
                    Operator::F64Store { .. } => LoadKind::F64,
                    Operator::I32Store { .. } => LoadKind::I32,
                    Operator::I64Store { .. } => LoadKind::I64,
                    Operator::I32Store8 { .. } => LoadKind::I8,
                    _ => LoadKind::I16,
                };
                let event = StoreEvent { addr, value, kind };
                if let Some(&li) = open_loops.last() {
                    facts.loops[li].stores.push(event.clone());
                }
                facts.all_stores.push(event);
            }
            // Integer arithmetic used in address/induction computation.
            Operator::I32Add | Operator::I64Add => bin(&mut stack, BinOp::IAdd),
            Operator::I32Sub | Operator::I64Sub => bin(&mut stack, BinOp::ISub),
            Operator::I32Mul | Operator::I64Mul => bin(&mut stack, BinOp::IMul),
            Operator::I32Shl | Operator::I64Shl => bin(&mut stack, BinOp::IShl),
            // Float arithmetic.
            Operator::F32Add | Operator::F64Add => bin(&mut stack, BinOp::FAdd),
            Operator::F32Sub | Operator::F64Sub => bin(&mut stack, BinOp::FSub),
            Operator::F32Mul | Operator::F64Mul => bin(&mut stack, BinOp::FMul),
            Operator::F32Div | Operator::F64Div => bin(&mut stack, BinOp::FDiv),
            Operator::F32Min | Operator::F64Min => bin(&mut stack, BinOp::FMin),
            Operator::F32Max | Operator::F64Max => bin(&mut stack, BinOp::FMax),
            // Comparisons.
            Operator::I32Eq
            | Operator::I32Ne
            | Operator::I32LtS
            | Operator::I32LtU
            | Operator::I32GtS
            | Operator::I32GtU
            | Operator::I32LeS
            | Operator::I32LeU
            | Operator::I32GeS
            | Operator::I32GeU
            | Operator::F32Eq
            | Operator::F32Ne
            | Operator::F32Lt
            | Operator::F32Gt
            | Operator::F32Le
            | Operator::F32Ge => bin(&mut stack, BinOp::Cmp),
            Operator::I32Eqz => {
                let v = pop(&mut stack);
                stack.push(Rc::new(Expr::Bin {
                    op: BinOp::Cmp,
                    lhs: v,
                    rhs: Rc::new(Expr::Const(0.0)),
                }));
            }
            // Conversions.
            Operator::I32TruncF32S
            | Operator::I32TruncF32U
            | Operator::I32TruncSatF32S
            | Operator::I32TruncSatF32U
            | Operator::I32TruncF64S
            | Operator::I32TruncSatF64S => {
                let v = pop(&mut stack);
                stack.push(Rc::new(Expr::Convert {
                    op: ConvertOp::F32ToI32,
                    value: v,
                }));
            }
            Operator::F32ConvertI32S
            | Operator::F32ConvertI32U
            | Operator::F64ConvertI32S
            | Operator::F32DemoteF64
            | Operator::F64PromoteF32 => {
                let v = pop(&mut stack);
                stack.push(Rc::new(Expr::Convert {
                    op: ConvertOp::I32ToF32,
                    value: v,
                }));
            }
            Operator::F32Neg
            | Operator::F64Neg
            | Operator::F32Abs
            | Operator::F64Abs
            | Operator::F32Sqrt
            | Operator::F64Sqrt
            | Operator::F32Nearest => {
                let v = pop(&mut stack);
                stack.push(Rc::new(Expr::Convert {
                    op: ConvertOp::Other,
                    value: v,
                }));
            }
            Operator::Nop | Operator::Unreachable => {}
            // Everything else: consume nothing, produce one unknown. This is
            // deliberately conservative — unmodeled ops cannot create
            // spurious structure, only hide it.
            _ => {
                stack.push(unknown());
            }
        }
    }
    facts
}

fn bin(stack: &mut Vec<Rc<Expr>>, op: BinOp) {
    let rhs = stack.pop().unwrap_or_else(|| Rc::new(Expr::Unknown));
    let lhs = stack.pop().unwrap_or_else(|| Rc::new(Expr::Unknown));
    stack.push(Rc::new(Expr::Bin { op, lhs, rhs }));
}
