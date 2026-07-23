#![cfg(all(not(target_arch = "wasm32"), feature = "bench-tool"))]

//! Benchmark integrity tests: the measurement core must execute real work,
//! consume results, gate on correctness, and record metadata. Timings from
//! incorrect implementations must never exist.

use barq_wasm::bench;

#[test]
fn incorrect_results_produce_no_timings() {
    let result = bench::measure(
        "fake_workload",
        "broken_backend",
        100,
        5,
        10,
        || false, // verification fails
        || 42u64,
    );
    assert!(!result.correct);
    assert!(
        result.stats.is_none(),
        "an incorrect implementation must never publish timings"
    );
}

#[test]
fn correct_results_have_real_nonzero_timings() {
    let data: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.5).collect();
    let result = bench::measure(
        "l2_norm_f32",
        "scalar",
        data.len(),
        3,
        20,
        || true,
        || barq_wasm::kernels::l2_norm_scalar(&data).unwrap(),
    );
    assert!(result.correct);
    let stats = result.stats.expect("stats for correct run");
    assert!(stats.min_ns > 0, "zero-duration samples are invalid");
    assert!(stats.median_ns >= stats.min_ns);
    assert!(stats.p95_ns >= stats.median_ns);
    assert!(stats.max_ns >= stats.p95_ns);
    assert_eq!(result.iterations, 20);
}

#[test]
fn environment_metadata_is_recorded() {
    let env = bench::environment();
    assert!(!env.os.is_empty());
    assert!(!env.arch.is_empty());
    assert!(!env.detected_features.is_empty());
    assert!(!env.selected_backend.is_empty());
    assert!(env.timestamp_unix > 1_700_000_000, "timestamp must be real");
}

#[test]
fn no_sleep_in_benchmark_sources() {
    // The old benchmarks timed thread::sleep. Guard against regression in
    // every benchmark-related source file.
    for path in [
        "src/bench/mod.rs",
        "src/bin/barq_bench.rs",
        "benches/kernels.rs",
        "docs/browser/benchmark.html",
    ] {
        let full = format!("{}/{path}", env!("CARGO_MANIFEST_DIR"));
        let source = std::fs::read_to_string(&full).expect(path);
        assert!(
            !source.contains("thread::sleep"),
            "{path} must not contain thread::sleep"
        );
        assert!(
            !source.contains("from_nanos("),
            "{path} must not fabricate durations"
        );
    }
}

#[test]
fn checked_in_benchmark_results_are_valid() {
    // Every checked-in benchmark JSON must parse, carry environment
    // metadata, contain no incorrect results, and no zero durations.
    let dir = format!("{}/benchmarks/results", env!("CARGO_MANIFEST_DIR"));
    let entries: Vec<_> = std::fs::read_dir(&dir)
        .map(|rd| rd.filter_map(|e| e.ok()).collect())
        .unwrap_or_default();
    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let report: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{path:?} is not valid JSON: {e}"));
        let env = &report["environment"];
        for key in ["os", "arch", "cpu", "git_commit", "selected_backend"] {
            assert!(
                env[key].is_string() && !env[key].as_str().unwrap().is_empty(),
                "{path:?}: environment.{key} missing"
            );
        }
        let results = report["results"].as_array().expect("results array");
        assert!(!results.is_empty());
        for r in results {
            assert_eq!(
                r["correct"], true,
                "{path:?}: incorrect result checked in: {r}"
            );
            let median = r["stats"]["median_ns"].as_u64().expect("median_ns");
            assert!(median > 0, "{path:?}: zero-duration sample in {r}");
        }
    }
}
