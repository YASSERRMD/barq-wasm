//! Abstract expression trees built by symbolically executing a function body.
//!
//! The analyzer never pattern-matches on raw byte counts; it reconstructs
//! what each operator computes in terms of locals, constants, loads, and
//! arithmetic, so pattern detection can reason about structure (e.g. "this
//! store writes `a[i] * b[i]`", "this local accumulates a product of two
//! loads from different base pointers").

use std::collections::BTreeSet;
use std::rc::Rc;

/// A symbolic value on the abstract operand stack.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Constant (any numeric type).
    Const(f64),
    /// Value of a local at read time.
    Local(u32),
    /// A load from linear memory at the given address expression.
    Load {
        addr: Rc<Expr>,
        kind: LoadKind,
    },
    /// Binary arithmetic.
    Bin {
        op: BinOp,
        lhs: Rc<Expr>,
        rhs: Rc<Expr>,
    },
    /// Numeric conversion (e.g. i32.trunc_sat_f32_s).
    Convert {
        op: ConvertOp,
        value: Rc<Expr>,
    },
    /// select(cond, a, b) or min/max-like constructs.
    Select {
        a: Rc<Expr>,
        b: Rc<Expr>,
    },
    /// Anything the interpreter does not model.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadKind {
    F32,
    F64,
    I32,
    I64,
    I8,
    I16,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    IAdd,
    ISub,
    IMul,
    IShl,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FMin,
    FMax,
    Cmp,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertOp {
    F32ToI32,
    I32ToF32,
    Other,
}

impl Expr {
    /// All locals appearing in this expression.
    pub fn locals(&self) -> BTreeSet<u32> {
        let mut set = BTreeSet::new();
        self.collect_locals(&mut set);
        set
    }

    fn collect_locals(&self, set: &mut BTreeSet<u32>) {
        match self {
            Expr::Local(i) => {
                set.insert(*i);
            }
            Expr::Load { addr, .. } => addr.collect_locals(set),
            Expr::Bin { lhs, rhs, .. } => {
                lhs.collect_locals(set);
                rhs.collect_locals(set);
            }
            Expr::Convert { value, .. } => value.collect_locals(set),
            Expr::Select { a, b } => {
                a.collect_locals(set);
                b.collect_locals(set);
            }
            Expr::Const(_) | Expr::Unknown => {}
        }
    }

    /// Depth-first traversal of all sub-expressions, including self.
    pub fn walk(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Load { addr, .. } => addr.walk(f),
            Expr::Bin { lhs, rhs, .. } => {
                lhs.walk(f);
                rhs.walk(f);
            }
            Expr::Convert { value, .. } => value.walk(f),
            Expr::Select { a, b } => {
                a.walk(f);
                b.walk(f);
            }
            _ => {}
        }
    }

    /// Collect every load in this expression.
    pub fn loads(&self) -> Vec<(Rc<Expr>, LoadKind)> {
        let mut out = Vec::new();
        self.walk(&mut |e| {
            if let Expr::Load { addr, kind } = e {
                out.push((addr.clone(), *kind));
            }
        });
        out
    }

    /// True if the expression contains a float multiply.
    pub fn contains_fmul(&self) -> bool {
        let mut found = false;
        self.walk(&mut |e| {
            if matches!(
                e,
                Expr::Bin {
                    op: BinOp::FMul,
                    ..
                }
            ) {
                found = true;
            }
        });
        found
    }
}
