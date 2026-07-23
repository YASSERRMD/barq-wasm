//! End-to-end tests for the barq-wasm CLI binary.

#![cfg(feature = "cli")]

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_barq-wasm"))
}

fn fixture(name: &str) -> String {
    format!("{}/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn run_add_wasm_returns_42() {
    let out = bin()
        .args([
            "run",
            &fixture("add.wasm"),
            "--invoke",
            "add",
            "--arg-i32",
            "20",
            "--arg-i32",
            "22",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "42");
}

#[test]
fn run_wat_module_works() {
    let out = bin()
        .args([
            "run",
            &fixture("add.wat"),
            "--invoke",
            "add",
            "--arg-i32",
            "-10",
            "--arg-i32",
            "52",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "42");
}

#[test]
fn validate_accepts_valid_module() {
    let out = bin()
        .args(["validate", &fixture("add.wasm")])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "valid");
}

#[test]
fn validate_rejects_garbage() {
    let dir = std::env::temp_dir().join("barq_cli_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("garbage.wasm");
    std::fs::write(&path, b"definitely not wasm").unwrap();
    let out = bin()
        .args(["validate", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("validation failed"));
}

#[test]
fn inspect_lists_exports() {
    let out = bin()
        .args(["inspect", &fixture("add.wasm")])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("add"), "inspect output: {stdout}");
}

#[test]
fn missing_file_exits_nonzero() {
    let out = bin()
        .args(["run", "no/such/file.wasm", "--invoke", "f"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));
}

#[test]
fn missing_export_exits_nonzero() {
    let out = bin()
        .args(["run", &fixture("add.wasm"), "--invoke", "nope"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("nope"));
}

#[test]
fn benchmark_reports_stats_and_result() {
    let out = bin()
        .args([
            "benchmark",
            &fixture("add.wasm"),
            "--invoke",
            "add",
            "--arg-i32",
            "20",
            "--arg-i32",
            "22",
            "--iterations",
            "20",
            "--warmup",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("result: 42"));
    assert!(stdout.contains("median:"));
    assert!(stdout.contains("p95:"));
}
