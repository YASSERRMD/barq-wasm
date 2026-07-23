//! xtask verify-all: run every verification layer and produce an honest
//! report. Each step ends PASS, FAIL, or NOT RUN (with the reason); NOT RUN
//! is never represented as PASS, and any FAIL makes the command exit
//! non-zero.
//!
//! Usage:
//!   cargo xtask verify-all [--with-browser]

use serde_json::json;
use std::process::{Command, ExitCode};

#[derive(Debug, Clone, PartialEq)]
enum Outcome {
    Pass,
    Fail(String),
    NotRun(String),
}

struct Step {
    name: &'static str,
    outcome: Outcome,
}

fn run(cmd: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("cannot spawn {cmd}: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if output.status.success() {
        Ok(stdout)
    } else {
        Err(format!(
            "{cmd} {} failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.join(" "),
            tail(&stdout, 30),
            tail(&stderr, 30)
        ))
    }
}

fn tail(s: &str, lines: usize) -> String {
    let all: Vec<&str> = s.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

fn step(name: &'static str, f: impl FnOnce() -> Outcome) -> Step {
    println!("==> {name}");
    let outcome = f();
    match &outcome {
        Outcome::Pass => println!("    PASS"),
        Outcome::Fail(e) => println!("    FAIL\n{e}"),
        Outcome::NotRun(why) => println!("    NOT RUN ({why})"),
    }
    Step { name, outcome }
}

fn cargo(args: &[&str]) -> Outcome {
    match run("cargo", args) {
        Ok(_) => Outcome::Pass,
        Err(e) => Outcome::Fail(e),
    }
}

fn script(path: &str) -> Outcome {
    match run("bash", &[path]) {
        Ok(_) => Outcome::Pass,
        Err(e) => Outcome::Fail(e),
    }
}

fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("verify-all") {
        eprintln!("usage: cargo xtask verify-all [--with-browser]");
        return ExitCode::FAILURE;
    }
    let with_browser = args.iter().any(|a| a == "--with-browser");

    // Detect host SIMD capabilities from a quick bench run's metadata.
    let caps = run(
        "cargo",
        &[
            "run",
            "-q",
            "--features",
            "bench-tool",
            "--bin",
            "barq-bench",
            "--",
            "--quick",
        ],
    )
    .ok()
    .and_then(|out| serde_json::from_str::<serde_json::Value>(&out).ok())
    .map(|v| v["environment"]["detected_features"].to_string())
    .unwrap_or_default();
    let host_avx2 = caps.contains("avx2: true");
    let host_neon = caps.contains("neon: true");

    let mut steps = vec![
        step("formatting (cargo fmt --check)", || {
            cargo(&["fmt", "--check"])
        }),
        step("lint (clippy -D warnings, all targets/features)", || {
            cargo(&[
                "clippy",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ])
        }),
        step("truthfulness gate (scripts/check-truthfulness.sh)", || {
            script("scripts/check-truthfulness.sh")
        }),
        step("native tests (default features)", || cargo(&["test"])),
        step("native tests (all features, incl. JIT)", || {
            cargo(&["test", "--all-features"])
        }),
        step("runtime integration tests", || {
            cargo(&["test", "--test", "runtime_integration"])
        }),
        step("analyzer precision tests", || {
            cargo(&["test", "--test", "analyzer_tests"])
        }),
        step("specialization differential tests", || {
            cargo(&["test", "--test", "specialization_tests"])
        }),
        step("JIT execution tests", || {
            cargo(&[
                "test",
                "--features",
                "jit-specialization",
                "--test",
                "jit_tests",
            ])
        }),
        step("fuzz smoke tests", || {
            cargo(&["test", "--test", "fuzz_smoke"])
        }),
        step("benchmark integrity tests", || {
            cargo(&[
                "test",
                "--features",
                "bench-tool",
                "--test",
                "bench_integrity",
            ])
        }),
        step("benchmark smoke run (real work, correctness-gated)", || {
            cargo(&[
                "run",
                "-q",
                "--release",
                "--features",
                "bench-tool",
                "--bin",
                "barq-bench",
                "--",
                "--quick",
                "--out",
                "target/verify-bench.json",
            ])
        }),
        step("native SIMD instruction verification", || {
            script("scripts/verify-native-simd.sh")
        }),
    ];

    // Architecture-specific differential coverage. The kernel tests run the
    // SIMD paths only when the host supports them; report that honestly.
    steps.push(step("x86 AVX2 differential execution", || {
        if host_avx2 {
            cargo(&["test", "--release", "--test", "kernel_differential"])
        } else {
            Outcome::NotRun("host CPU has no AVX2; runs on x86-64 CI".to_string())
        }
    }));
    steps.push(step("ARM NEON differential execution", || {
        if host_neon {
            cargo(&["test", "--release", "--test", "kernel_differential"])
        } else {
            Outcome::NotRun("host CPU has no NEON".to_string())
        }
    }));

    steps.push(step(
        "browser bundles build + wasm SIMD verification",
        || {
            if have("wasm-pack") {
                script("scripts/verify-wasm-simd.sh")
            } else {
                Outcome::NotRun("wasm-pack not installed".to_string())
            }
        },
    ));
    steps.push(step("browser correctness tests (headless Chrome)", || {
        if !with_browser {
            Outcome::NotRun("pass --with-browser to run".to_string())
        } else if !have("wasm-pack") {
            Outcome::NotRun("wasm-pack not installed".to_string())
        } else {
            let simd = Command::new("wasm-pack")
                .env("RUSTFLAGS", "-C target-feature=+simd128")
                .args([
                    "test",
                    "--headless",
                    "--chrome",
                    "--no-default-features",
                    "--features",
                    "browser",
                ])
                .status();
            match simd {
                Ok(s) if s.success() => Outcome::Pass,
                Ok(_) => Outcome::Fail("browser tests failed".to_string()),
                Err(e) => Outcome::Fail(e.to_string()),
            }
        }
    }));
    steps.push(step("documentation build (cargo doc)", || {
        cargo(&["doc", "--no-deps", "--all-features"])
    }));

    // ---- report ----
    let mut failed = false;
    let mut md = String::from("# Barq-WASM verification\n\n");
    let mut results = vec![];
    for s in &steps {
        let (status, detail) = match &s.outcome {
            Outcome::Pass => ("PASS", String::new()),
            Outcome::Fail(e) => {
                failed = true;
                ("FAIL", e.clone())
            }
            Outcome::NotRun(why) => ("NOT RUN", why.clone()),
        };
        md.push_str(&format!(
            "- {}: **{}**{}\n",
            s.name,
            status,
            if detail.is_empty() || status == "FAIL" {
                String::new()
            } else {
                format!(" — {detail}")
            }
        ));
        results.push(json!({
            "step": s.name,
            "status": status,
            "detail": detail,
        }));
    }
    let report = json!({
        "detected_features": caps,
        "steps": results,
        "overall": if failed { "FAIL" } else { "PASS" },
    });
    std::fs::write(
        "verification-report.json",
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
    md.push_str(&format!(
        "\nOverall: **{}**\n",
        if failed { "FAIL" } else { "PASS" }
    ));
    std::fs::write("verification-report.md", &md).unwrap();

    println!("\n{md}");
    println!("Reports written: verification-report.json, verification-report.md");
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
