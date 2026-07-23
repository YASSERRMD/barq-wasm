//! Dynamically-typed WebAssembly values for `invoke_dynamic`.

use crate::error::{BarqError, BarqResult};
use wasmtime::Val;

/// A core WebAssembly numeric value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WasmValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl WasmValue {
    pub fn to_val(self) -> Val {
        match self {
            WasmValue::I32(v) => Val::I32(v),
            WasmValue::I64(v) => Val::I64(v),
            WasmValue::F32(v) => Val::F32(v.to_bits()),
            WasmValue::F64(v) => Val::F64(v.to_bits()),
        }
    }

    pub fn from_val(val: &Val) -> BarqResult<Self> {
        match val {
            Val::I32(v) => Ok(WasmValue::I32(*v)),
            Val::I64(v) => Ok(WasmValue::I64(*v)),
            Val::F32(bits) => Ok(WasmValue::F32(f32::from_bits(*bits))),
            Val::F64(bits) => Ok(WasmValue::F64(f64::from_bits(*bits))),
            other => Err(BarqError::UnsupportedFeature(format!(
                "non-numeric wasm value in results: {other:?}"
            ))),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            WasmValue::I32(_) => "i32",
            WasmValue::I64(_) => "i64",
            WasmValue::F32(_) => "f32",
            WasmValue::F64(_) => "f64",
        }
    }
}

impl std::fmt::Display for WasmValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmValue::I32(v) => write!(f, "{v}"),
            WasmValue::I64(v) => write!(f, "{v}"),
            WasmValue::F32(v) => write!(f, "{v}"),
            WasmValue::F64(v) => write!(f, "{v}"),
        }
    }
}
