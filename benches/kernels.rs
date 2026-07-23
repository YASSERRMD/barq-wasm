//! Criterion benchmarks over the real kernels.
//!
//! Correctness is asserted against the scalar reference before any timing;
//! inputs are preallocated outside the timed region; results are consumed
//! with black_box. There are no sleeps and no synthetic numbers.

use barq_wasm::kernels;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

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

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= tol * a.abs().max(b.abs()).max(1.0),
        "{ctx}: {a} vs {b}"
    );
}

fn bench_dot_product(c: &mut Criterion) {
    let caps = kernels::cpu_capabilities();
    let mut group = c.benchmark_group("dot_product_f32");
    for &size in &[16usize, 128, 1024, 16_384, 100_000, 1_000_000] {
        let a = pseudo_random_f32(size, 1);
        let b = pseudo_random_f32(size, 2);
        let reference = kernels::dot_product_scalar(&a, &b).unwrap();

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| black_box(kernels::dot_product_scalar(black_box(&a), black_box(&b))))
        });

        // Correctness gate before timing the optimized paths.
        assert_close(
            barq_wasm::wasm_bindings::dot_product_unrolled_scalar(&a, &b),
            reference,
            1e-3,
            "unrolled",
        );
        group.bench_with_input(
            BenchmarkId::new("unrolled_scalar", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    black_box(barq_wasm::wasm_bindings::dot_product_unrolled_scalar(
                        black_box(&a),
                        black_box(&b),
                    ))
                })
            },
        );

        assert_close(
            kernels::dot_product(&a, &b).unwrap(),
            reference,
            1e-3,
            "auto",
        );
        group.bench_with_input(BenchmarkId::new("auto", size), &size, |bench, _| {
            bench.iter(|| black_box(kernels::dot_product(black_box(&a), black_box(&b))))
        });

        if caps.avx2 && caps.fma {
            assert_close(
                kernels::dot_product_avx2_fma(&a, &b).unwrap(),
                reference,
                1e-3,
                "avx2_fma",
            );
            group.bench_with_input(BenchmarkId::new("avx2_fma", size), &size, |bench, _| {
                bench
                    .iter(|| black_box(kernels::dot_product_avx2_fma(black_box(&a), black_box(&b))))
            });
        }
        if caps.neon {
            assert_close(
                kernels::dot_product_neon(&a, &b).unwrap(),
                reference,
                1e-3,
                "neon",
            );
            group.bench_with_input(BenchmarkId::new("neon", size), &size, |bench, _| {
                bench.iter(|| black_box(kernels::dot_product_neon(black_box(&a), black_box(&b))))
            });
        }
    }
    group.finish();
}

fn bench_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_f32_to_i8");
    for &size in &[1024usize, 100_000, 500_000] {
        let input = pseudo_random_f32(size, 5);
        let reference = kernels::quantize_i8_scalar(&input, 0.5).unwrap();
        assert_eq!(
            reference,
            kernels::quantize_i8(&input, 0.5).unwrap(),
            "quantize auto must be bit-exact before timing"
        );
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| black_box(kernels::quantize_i8_scalar(black_box(&input), 0.5)))
        });
        group.bench_with_input(BenchmarkId::new("auto", size), &size, |bench, _| {
            bench.iter(|| black_box(kernels::quantize_i8(black_box(&input), 0.5)))
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiply_f32");
    group.sample_size(20);
    for &n in &[16usize, 32, 64, 128, 256] {
        let a = pseudo_random_f32(n * n, 7);
        let b = pseudo_random_f32(n * n, 8);
        let reference = kernels::matrix_multiply_scalar(&a, &b, n, n, n).unwrap();
        let auto = kernels::matrix_multiply(&a, &b, n, n, n).unwrap();
        for i in 0..n * n {
            assert_close(auto[i], reference[i], 1e-3, "matmul auto");
        }
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(kernels::matrix_multiply_scalar(
                    black_box(&a),
                    black_box(&b),
                    n,
                    n,
                    n,
                ))
            })
        });
        group.bench_with_input(BenchmarkId::new("auto", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(kernels::matrix_multiply(
                    black_box(&a),
                    black_box(&b),
                    n,
                    n,
                    n,
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dot_product, bench_quantize, bench_matmul);
criterion_main!(benches);
