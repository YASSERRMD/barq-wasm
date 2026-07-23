//! Structural WebAssembly module analysis.
//!
//! Parses real modules with `wasmparser` (types, imports, exports, code) and
//! symbolically executes each function body to detect computational patterns
//! from structural evidence: loop nesting, load/store address expressions,
//! accumulation, induction variables, conversions. Confidence is always
//! derived from satisfied requirements and every candidate carries its
//! evidence; nothing is inferred from raw opcode counts.

mod engine;
pub mod expr;
mod patterns;

pub use engine::{BodyFacts, LoopFacts};
pub use patterns::{PatternCandidate, PatternEvidence, PatternKind};

use crate::error::{BarqError, BarqResult};
use wasmparser::{Parser, Payload};

/// Analysis of one function body.
#[derive(Debug)]
pub struct FunctionAnalysis {
    /// Module-wide function index (imports included).
    pub function_index: u32,
    /// Export name, when this function is exported.
    pub export_name: Option<String>,
    pub operator_count: usize,
    pub loop_count: usize,
    pub max_loop_depth: usize,
    pub candidates: Vec<PatternCandidate>,
}

/// Analysis of one import.
#[derive(Debug)]
pub struct ImportAnalysis {
    pub module: String,
    pub name: String,
}

/// Analysis of one export.
#[derive(Debug)]
pub struct ExportAnalysis {
    pub name: String,
    pub function_index: Option<u32>,
}

/// Whole-module structural analysis.
#[derive(Debug)]
pub struct ModuleAnalysis {
    pub functions: Vec<FunctionAnalysis>,
    pub imports: Vec<ImportAnalysis>,
    pub exports: Vec<ExportAnalysis>,
}

impl ModuleAnalysis {
    /// All candidates across functions, tagged with their function.
    pub fn candidates(&self) -> impl Iterator<Item = (&FunctionAnalysis, &PatternCandidate)> {
        self.functions
            .iter()
            .flat_map(|f| f.candidates.iter().map(move |c| (f, c)))
    }
}

/// Analyze a binary WebAssembly module.
///
/// The bytes must be a valid binary module (not WAT text); parse failures
/// return `BarqError::Validation`.
pub fn analyze_module(bytes: &[u8]) -> BarqResult<ModuleAnalysis> {
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut imported_funcs = 0u32;
    let mut code_index = 0u32;
    let mut functions = Vec::new();

    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload.map_err(|e| BarqError::Validation(e.to_string()))?;
        match payload {
            Payload::ImportSection(reader) => {
                for import in reader {
                    let import = import.map_err(|e| BarqError::Validation(e.to_string()))?;
                    if matches!(import.ty, wasmparser::TypeRef::Func(_)) {
                        imported_funcs += 1;
                    }
                    imports.push(ImportAnalysis {
                        module: import.module.to_string(),
                        name: import.name.to_string(),
                    });
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.map_err(|e| BarqError::Validation(e.to_string()))?;
                    exports.push(ExportAnalysis {
                        name: export.name.to_string(),
                        function_index: matches!(export.kind, wasmparser::ExternalKind::Func)
                            .then_some(export.index),
                    });
                }
            }
            Payload::CodeSectionEntry(body) => {
                let ops: Vec<_> = body
                    .get_operators_reader()
                    .map_err(|e| BarqError::Validation(e.to_string()))?
                    .into_iter()
                    .collect::<Result<_, _>>()
                    .map_err(|e| BarqError::Validation(e.to_string()))?;
                let facts = engine::analyze_body(&ops);

                let mut candidates = Vec::new();
                for (i, lp) in facts.loops.iter().enumerate() {
                    if lp.innermost {
                        if let Some(c) = patterns::detect_dot_product(lp, i) {
                            candidates.push(c);
                        }
                    }
                    if let Some(c) = patterns::detect_quantization(lp, i) {
                        candidates.push(c);
                    }
                }
                if let Some(c) = patterns::detect_matrix_multiply(&facts) {
                    // A matmul body contains a dot-product-shaped inner loop;
                    // prefer the more specific whole-function candidate.
                    candidates.retain(|d| d.pattern != PatternKind::DotProduct);
                    candidates.push(c);
                }

                functions.push(FunctionAnalysis {
                    function_index: imported_funcs + code_index,
                    export_name: None,
                    operator_count: facts.operator_count,
                    loop_count: facts.loops.len(),
                    max_loop_depth: facts.max_loop_depth,
                    candidates,
                });
                code_index += 1;
            }
            _ => {}
        }
    }

    if functions.is_empty() && imports.is_empty() && exports.is_empty() {
        return Err(BarqError::Validation(
            "not a WebAssembly module (no sections parsed)".to_string(),
        ));
    }

    // Attach export names to functions.
    for f in &mut functions {
        f.export_name = exports
            .iter()
            .find(|e| e.function_index == Some(f.function_index))
            .map(|e| e.name.clone());
    }

    Ok(ModuleAnalysis {
        functions,
        imports,
        exports,
    })
}
