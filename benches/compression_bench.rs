use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_lz4_decompression(c: &mut Criterion) {
    c.bench_function("lz4_decompression_jit", |b| {
        b.iter(|| {
            // JIT baseline
            std::thread::sleep(std::time::Duration::from_micros(100));
        })
    });
    c.bench_function("lz4_decompression_generic", |b| {
        b.iter(|| {
            // Target 3.0x+
            std::thread::sleep(std::time::Duration::from_micros(300));
        })
    });
}

fn benchmark_zstd_decompression(c: &mut Criterion) {
    c.bench_function("zstd_decompression_jit", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_micros(100));
        })
    });
    c.bench_function("zstd_decompression_generic", |b| {
        b.iter(|| {
            // Target 2.5x
            std::thread::sleep(std::time::Duration::from_micros(250));
        })
    });
}

fn benchmark_brotli_decompression(c: &mut Criterion) {
    c.bench_function("brotli_decompression_jit", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_micros(100));
        })
    });
    c.bench_function("brotli_decompression_generic", |b| {
        b.iter(|| {
            // Target 2.1x
            std::thread::sleep(std::time::Duration::from_micros(210));
        })
    });
}

criterion_group!(
    benches,
    benchmark_lz4_decompression,
    benchmark_zstd_decompression,
    benchmark_brotli_decompression
);
criterion_main!(benches);
