//! wasm-inspect: verify the instruction content and exports of a .wasm file.
//!
//! Used by scripts/verify-wasm-simd.sh to prove that the "simd" browser
//! bundle really contains SIMD instructions and the "scalar" bundle contains
//! none. Exits non-zero when an expectation fails.
//!
//! Usage:
//!   wasm-inspect <file.wasm> [--expect-simd] [--expect-no-simd]
//!                [--require-export NAME]... [--forbid-export NAME]...
//!                [--require-op OPNAME]...

use std::collections::BTreeMap;
use std::process::ExitCode;
use wasmparser::{Operator, Parser, Payload};

fn op_name(op: &Operator) -> String {
    let dbg = format!("{op:?}");
    dbg.split(&[' ', '{'][..]).next().unwrap_or("").to_string()
}

fn is_simd_op(name: &str) -> bool {
    name.starts_with("V128")
        || name.starts_with("F32x4")
        || name.starts_with("F64x2")
        || name.starts_with("I8x16")
        || name.starts_with("I16x8")
        || name.starts_with("I32x4")
        || name.starts_with("I64x2")
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: wasm-inspect <file.wasm> [flags]");
        return ExitCode::FAILURE;
    }
    let path = &args[0];
    let mut expect_simd = false;
    let mut expect_no_simd = false;
    let mut require_exports: Vec<String> = vec![];
    let mut forbid_exports: Vec<String> = vec![];
    let mut require_ops: Vec<String> = vec![];
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--expect-simd" => expect_simd = true,
            "--expect-no-simd" => expect_no_simd = true,
            "--require-export" => {
                i += 1;
                require_exports.push(args[i].clone());
            }
            "--forbid-export" => {
                i += 1;
                forbid_exports.push(args[i].clone());
            }
            "--require-op" => {
                i += 1;
                require_ops.push(args[i].clone());
            }
            other => {
                eprintln!("unknown flag: {other}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut simd_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut exports: Vec<String> = vec![];
    let mut total_ops = 0usize;

    for payload in Parser::new(0).parse_all(&bytes) {
        match payload {
            Ok(Payload::ExportSection(reader)) => {
                for export in reader {
                    match export {
                        Ok(e) => exports.push(e.name.to_string()),
                        Err(e) => {
                            eprintln!("export parse error: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
            Ok(Payload::CodeSectionEntry(body)) => {
                let mut reader = match body.get_operators_reader() {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("code parse error: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                while !reader.eof() {
                    match reader.read() {
                        Ok(op) => {
                            total_ops += 1;
                            let name = op_name(&op);
                            if is_simd_op(&name) {
                                *simd_counts.entry(name).or_insert(0) += 1;
                            }
                        }
                        Err(e) => {
                            eprintln!("operator parse error: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("parse error: {e}");
                return ExitCode::FAILURE;
            }
            _ => {}
        }
    }

    let simd_total: usize = simd_counts.values().sum();
    println!("file: {path}");
    println!("total operators: {total_ops}");
    println!("simd operators: {simd_total}");
    for (name, count) in &simd_counts {
        println!("  {name}: {count}");
    }

    let mut fail = false;
    if expect_simd && simd_total == 0 {
        eprintln!("FAIL: expected SIMD instructions, found none");
        fail = true;
    }
    if expect_no_simd && simd_total > 0 {
        eprintln!("FAIL: expected no SIMD instructions, found {simd_total}");
        fail = true;
    }
    for op in &require_ops {
        if !simd_counts.contains_key(op) {
            eprintln!("FAIL: required instruction '{op}' not present");
            fail = true;
        } else {
            println!("required instruction present: {op}");
        }
    }
    for name in &require_exports {
        if !exports.iter().any(|e| e == name) {
            eprintln!("FAIL: required export '{name}' missing");
            fail = true;
        } else {
            println!("required export present: {name}");
        }
    }
    for name in &forbid_exports {
        if exports.iter().any(|e| e == name) {
            eprintln!("FAIL: export '{name}' must not be present in this bundle");
            fail = true;
        }
    }

    if fail {
        ExitCode::FAILURE
    } else {
        println!("OK");
        ExitCode::SUCCESS
    }
}
