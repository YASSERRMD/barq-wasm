//! Benchmark measurement core.
//!
//! Rules enforced here:
//! - Correctness first: every candidate's output is compared against the
//!   scalar reference *before* any timing. An incorrect implementation gets
//!   `correct: false` and NO timing numbers.
//! - Real work only: the measured closure's result is consumed via
//!   `std::hint::black_box`; there are no sleeps anywhere.
//! - Reproducibility: results carry environment metadata (OS, arch, CPU,
//!   detected features, git commit, sample counts, input sizes).

use crate::kernels;
use serde::Serialize;
use std::hint::black_box;
use std::time::Instant;

/// Statistics over one timed workload, in nanoseconds.
#[derive(Debug, Clone, Serialize)]
pub struct BenchStats {
    pub min_ns: u128,
    pub median_ns: u128,
    pub p90_ns: u128,
    pub p95_ns: u128,
    pub max_ns: u128,
    pub mean_ns: f64,
    pub stddev_ns: f64,
}

/// One benchmark record.
#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub workload: String,
    pub backend: String,
    pub size: usize,
    pub warmup_iterations: usize,
    pub iterations: usize,
    pub correct: bool,
    /// Present only when `correct` is true.
    pub stats: Option<BenchStats>,
}

/// Environment metadata recorded with every run.
#[derive(Debug, Clone, Serialize)]
pub struct BenchEnvironment {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub detected_features: String,
    pub selected_backend: String,
    pub git_commit: String,
    pub crate_version: String,
    pub timestamp_unix: u64,
}

/// A full benchmark report (what gets serialized to JSON).
#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub environment: BenchEnvironment,
    pub results: Vec<BenchResult>,
}

pub fn environment() -> BenchEnvironment {
    let caps = kernels::cpu_capabilities();
    let backend = kernels::select_backend()
        .map(|b| b.to_string())
        .unwrap_or_else(|e| format!("error: {e}"));
    BenchEnvironment {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu: cpu_brand(),
        detected_features: format!("{caps:?}"),
        selected_backend: backend,
        git_commit: command_output("git", &["rev-parse", "HEAD"]),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    }
}

fn command_output(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn cpu_brand() -> String {
    #[cfg(target_os = "macos")]
    {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".to_string()
    }
}

fn stats_from(mut samples_ns: Vec<u128>) -> BenchStats {
    samples_ns.sort_unstable();
    let n = samples_ns.len();
    let mean = samples_ns.iter().sum::<u128>() as f64 / n as f64;
    let variance = samples_ns
        .iter()
        .map(|&s| {
            let d = s as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let pick = |q: usize| samples_ns[(n * q / 100).min(n - 1)];
    BenchStats {
        min_ns: samples_ns[0],
        median_ns: samples_ns[n / 2],
        p90_ns: pick(90),
        p95_ns: pick(95),
        max_ns: samples_ns[n - 1],
        mean_ns: mean,
        stddev_ns: variance.sqrt(),
    }
}

/// Measure one workload.
///
/// `verify` is called once before timing; if it returns false, no timing
/// happens and the result is marked incorrect.
pub fn measure<T>(
    workload: &str,
    backend: &str,
    size: usize,
    warmup: usize,
    iterations: usize,
    verify: impl FnOnce() -> bool,
    mut run: impl FnMut() -> T,
) -> BenchResult {
    if !verify() {
        return BenchResult {
            workload: workload.to_string(),
            backend: backend.to_string(),
            size,
            warmup_iterations: warmup,
            iterations,
            correct: false,
            stats: None,
        };
    }
    for _ in 0..warmup {
        black_box(run());
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(run());
        samples.push(start.elapsed().as_nanos());
    }
    BenchResult {
        workload: workload.to_string(),
        backend: backend.to_string(),
        size,
        warmup_iterations: warmup,
        iterations,
        correct: true,
        stats: Some(stats_from(samples)),
    }
}

/// Relative-error check used by the float workload verifications.
pub fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}
