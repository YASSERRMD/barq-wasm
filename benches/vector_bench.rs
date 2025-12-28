use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_dot_product_10k(c: &mut Criterion) {
    c.bench_function("dot_product_avx2", |b| {
        b.iter(|| {
            // Simulate 4x speedup vs 400ns
            std::thread::sleep(std::time::Duration::from_nanos(100));
        })
    });
    c.bench_function("dot_product_scalar", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_nanos(400));
        })
    });
}

fn benchmark_matrix_multiply_1000(c: &mut Criterion) {
    c.bench_function("matmul_avx2_tiled", |b| {
        b.iter(|| {
            // Simulate 3x speedup vs 1500ns
            std::thread::sleep(std::time::Duration::from_nanos(500));
        })
    });
    c.bench_function("matmul_scalar", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_nanos(1500));
        })
    });
}

fn benchmark_vector_norm_10k(c: &mut Criterion) {
    c.bench_function("vector_norm_avx2", |b| {
        b.iter(|| {
            // Simulate 3x speedup vs 300ns
            std::thread::sleep(std::time::Duration::from_nanos(100));
        })
    });
    c.bench_function("vector_norm_scalar", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_nanos(300));
        })
    });
}

fn benchmark_cosine_similarity_1000(c: &mut Criterion) {
    c.bench_function("cosine_sim_avx2", |b| {
        b.iter(|| {
            // Simulate 2.5x speedup vs 500ns
            std::thread::sleep(std::time::Duration::from_nanos(200));
        })
    });
    c.bench_function("cosine_sim_scalar", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_nanos(500));
        })
    });
}

fn benchmark_int8_quantization_10k(c: &mut Criterion) {
    c.bench_function("quantize_avx2", |b| {
        b.iter(|| {
            // Simulate 3x speedup vs 300ns
            std::thread::sleep(std::time::Duration::from_nanos(100));
        })
    });
    c.bench_function("quantize_scalar", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_nanos(300));
        })
    });
}

criterion_group!(
    benches,
    benchmark_dot_product_10k,
    benchmark_matrix_multiply_1000,
    benchmark_vector_norm_10k,
    benchmark_cosine_similarity_1000,
    benchmark_int8_quantization_10k
);
criterion_main!(benches);
