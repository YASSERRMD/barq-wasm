//! barq-bench: reproducible native kernel benchmarks with correctness gates.
//!
//! Writes a JSON report (environment + per-workload statistics). Exits
//! non-zero if any candidate produced an incorrect result — a fast but wrong
//! implementation never publishes a timing.
//!
//! Usage:
//!   barq-bench [--out results.json] [--iterations N] [--warmup N] [--quick]

use barq_wasm::bench::{self, BenchReport, BenchResult};
use barq_wasm::kernels;
use std::process::ExitCode;

fn pseudo_random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 33) as f64 / (1u64 << 31) as f64) * 200.0 - 100.0) as f32
        })
        .collect()
}

struct Config {
    out: Option<String>,
    warmup: usize,
    iterations: usize,
    sizes: Vec<usize>,
    matrix_sizes: Vec<usize>,
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config {
        out: None,
        warmup: 20,
        iterations: 100,
        sizes: vec![
            15, 16, 17, 127, 128, 129, 1024, 16_384, 100_000, 500_000, 1_000_000,
        ],
        matrix_sizes: vec![16, 32, 64, 128, 256],
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                config.out = Some(args.get(i).ok_or("--out needs a path")?.clone());
            }
            "--iterations" => {
                i += 1;
                config.iterations = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or("--iterations needs a number")?;
            }
            "--warmup" => {
                i += 1;
                config.warmup = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or("--warmup needs a number")?;
            }
            "--quick" => {
                config.sizes = vec![16, 1024, 100_000];
                config.matrix_sizes = vec![16, 64];
                config.warmup = 2;
                config.iterations = 5;
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    Ok(config)
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let caps = kernels::cpu_capabilities();
    let mut results: Vec<BenchResult> = Vec::new();

    // ---- dot product / l2 norm / cosine over the size sweep ----
    for &size in &config.sizes {
        let a = pseudo_random_f32(size, 1);
        let b = pseudo_random_f32(size, 2);
        let reference = kernels::dot_product_scalar(&a, &b).unwrap();

        results.push(bench::measure(
            "dot_product_f32",
            "scalar",
            size,
            config.warmup,
            config.iterations,
            || true,
            || kernels::dot_product_scalar(&a, &b).unwrap(),
        ));
        results.push(bench::measure(
            "dot_product_f32",
            "unrolled_scalar",
            size,
            config.warmup,
            config.iterations,
            || {
                bench::close(
                    barq_wasm::wasm_bindings::dot_product_unrolled_scalar(&a, &b),
                    reference,
                    1e-3,
                )
            },
            || barq_wasm::wasm_bindings::dot_product_unrolled_scalar(&a, &b),
        ));
        let auto = kernels::dot_product_ex(&a, &b).unwrap();
        results.push(bench::measure(
            "dot_product_f32",
            &format!("auto:{}", auto.backend),
            size,
            config.warmup,
            config.iterations,
            || bench::close(auto.value, reference, 1e-3),
            || kernels::dot_product(&a, &b).unwrap(),
        ));
        if caps.avx2 {
            let v = kernels::dot_product_avx2(&a, &b).unwrap();
            results.push(bench::measure(
                "dot_product_f32",
                "avx2",
                size,
                config.warmup,
                config.iterations,
                || bench::close(v, reference, 1e-3),
                || kernels::dot_product_avx2(&a, &b).unwrap(),
            ));
        }
        if caps.avx2 && caps.fma {
            let v = kernels::dot_product_avx2_fma(&a, &b).unwrap();
            results.push(bench::measure(
                "dot_product_f32",
                "avx2_fma",
                size,
                config.warmup,
                config.iterations,
                || bench::close(v, reference, 1e-3),
                || kernels::dot_product_avx2_fma(&a, &b).unwrap(),
            ));
        }
        if caps.neon {
            let v = kernels::dot_product_neon(&a, &b).unwrap();
            results.push(bench::measure(
                "dot_product_f32",
                "neon",
                size,
                config.warmup,
                config.iterations,
                || bench::close(v, reference, 1e-3),
                || kernels::dot_product_neon(&a, &b).unwrap(),
            ));
        }

        // L2 norm scalar vs auto.
        let norm_ref = kernels::l2_norm_scalar(&a).unwrap();
        results.push(bench::measure(
            "l2_norm_f32",
            "scalar",
            size,
            config.warmup,
            config.iterations,
            || true,
            || kernels::l2_norm_scalar(&a).unwrap(),
        ));
        let norm_auto = kernels::l2_norm_ex(&a).unwrap();
        results.push(bench::measure(
            "l2_norm_f32",
            &format!("auto:{}", norm_auto.backend),
            size,
            config.warmup,
            config.iterations,
            || bench::close(norm_auto.value, norm_ref, 1e-3),
            || kernels::l2_norm(&a).unwrap(),
        ));

        // Quantization scalar vs auto (bit-exact requirement).
        let q_ref = kernels::quantize_i8_scalar(&a, 0.5).unwrap();
        results.push(bench::measure(
            "quantize_f32_to_i8",
            "scalar",
            size,
            config.warmup,
            config.iterations,
            || true,
            || kernels::quantize_i8_scalar(&a, 0.5).unwrap(),
        ));
        let q_auto = kernels::quantize_i8_ex(&a, 0.5).unwrap();
        results.push(bench::measure(
            "quantize_f32_to_i8",
            &format!("auto:{}", q_auto.backend),
            size,
            config.warmup,
            config.iterations,
            || q_auto.value == q_ref,
            || kernels::quantize_i8(&a, 0.5).unwrap(),
        ));
    }

    // ---- matrix multiply over square sizes ----
    for &n in &config.matrix_sizes {
        let a = pseudo_random_f32(n * n, 3);
        let b = pseudo_random_f32(n * n, 4);
        let reference = kernels::matrix_multiply_scalar(&a, &b, n, n, n).unwrap();
        // Different (but both valid) summation orders diverge with the
        // reduction length; bound the error by k * eps * max|a| * max|b|.
        let amax = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let bmax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let abs_tol = n as f32 * f32::EPSILON * amax * bmax * 8.0;
        results.push(bench::measure(
            "matrix_multiply_f32",
            "scalar",
            n,
            config.warmup,
            config.iterations.min(30),
            || true,
            || kernels::matrix_multiply_scalar(&a, &b, n, n, n).unwrap(),
        ));
        let auto = kernels::matrix_multiply_ex(&a, &b, n, n, n).unwrap();
        let auto_ok = auto
            .value
            .iter()
            .zip(reference.iter())
            .all(|(x, y)| (x - y).abs() <= abs_tol || bench::close(*x, *y, 1e-4));
        results.push(bench::measure(
            "matrix_multiply_f32",
            &format!("auto:{}", auto.backend),
            n,
            config.warmup,
            config.iterations.min(30),
            || auto_ok,
            || kernels::matrix_multiply(&a, &b, n, n, n).unwrap(),
        ));
    }

    let report = BenchReport {
        environment: bench::environment(),
        results,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize");
    match &config.out {
        Some(path) => {
            if let Err(e) = std::fs::write(path, &json) {
                eprintln!("error: cannot write {path}: {e}");
                return ExitCode::FAILURE;
            }
            println!("wrote {path}");
        }
        None => println!("{json}"),
    }

    let incorrect: Vec<_> = report
        .results
        .iter()
        .filter(|r| !r.correct)
        .map(|r| format!("{} [{}] size={}", r.workload, r.backend, r.size))
        .collect();
    if !incorrect.is_empty() {
        eprintln!("INCORRECT RESULTS (no timings published): {incorrect:?}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
