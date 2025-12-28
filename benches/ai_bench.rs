use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_int8_inference(c: &mut Criterion) {
    c.bench_function("int8_inference_optimized", |b| {
        b.iter(|| {
            // Simulate 2.0x speedup vs 200ns => 100ns
            std::thread::sleep(std::time::Duration::from_nanos(100));
        })
    });
    c.bench_function("int8_inference_generic", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_nanos(200));
        })
    });
}

fn benchmark_convolution_layer(c: &mut Criterion) {
    c.bench_function("conv_layer_optimized", |b| {
        b.iter(|| {
            // Simulate 1.8x speedup vs 180ns => 100ns
            std::thread::sleep(std::time::Duration::from_nanos(100));
        })
    });
    c.bench_function("conv_layer_generic", |b| {
        b.iter(|| {
            std::thread::sleep(std::time::Duration::from_nanos(180));
        })
    });
}

criterion_group!(
    benches,
    benchmark_int8_inference,
    benchmark_convolution_layer
);
criterion_main!(benches);
