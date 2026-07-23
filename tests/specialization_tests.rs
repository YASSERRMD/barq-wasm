#![cfg(all(not(target_arch = "wasm32"), feature = "native-runtime"))]

//! Barq host-kernel ABI tests: guests that import `barq.*` get real,
//! verified kernels, and results are differentially compared against
//! pure-guest scalar implementations of the same computation.

use barq_wasm::error::BarqError;
use barq_wasm::kernels;
use barq_wasm::runtime::{Runtime, RuntimeConfig, WasmValue};

fn runtime() -> Runtime {
    Runtime::new(RuntimeConfig::default()).expect("runtime")
}

fn load(rt: &mut Runtime, wat: &str) {
    let bytes = wat::parse_str(wat).expect("WAT must assemble");
    rt.load_module(&bytes).expect("load");
    rt.instantiate().expect("instantiate");
}

/// Guest module with both the barq import and its own scalar dot product,
/// so host-kernel and pure-guest results can be compared on the same data.
const DOT_MODULE: &str = r#"(module
  (import "barq" "dot_product_f32" (func $barq_dot (param i32 i32 i32) (result f32)))
  (memory (export "memory") 4)
  (func (export "dot_host") (param $a i32) (param $b i32) (param $n i32) (result f32)
    (call $barq_dot (local.get $a) (local.get $b) (local.get $n)))
  (func (export "dot_guest") (param $a i32) (param $b i32) (param $n i32) (result f32)
    (local $i i32) (local $sum f32)
    (block $done
      (loop $l
        (br_if $done (i32.ge_s (local.get $i) (local.get $n)))
        (local.set $sum (f32.add (local.get $sum)
          (f32.mul
            (f32.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2))))
            (f32.load (i32.add (local.get $b) (i32.shl (local.get $i) (i32.const 2)))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $l)))
    (local.get $sum)))"#;

fn write_f32s(rt: &mut Runtime, offset: usize, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    rt.write_memory(offset, &bytes).expect("write");
}

fn pseudo_random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 33) as f64 / (1u64 << 31) as f64) * 20.0 - 10.0) as f32
        })
        .collect()
}

#[test]
fn barq_dot_product_matches_pure_guest_and_native_scalar() {
    let mut rt = runtime();
    load(&mut rt, DOT_MODULE);
    for len in [0usize, 1, 7, 8, 64, 1000, 10_000] {
        let a = pseudo_random_f32(len, 3);
        let b = pseudo_random_f32(len, 4);
        let a_ptr = 1024usize;
        let b_ptr = a_ptr + len * 4;
        write_f32s(&mut rt, a_ptr, &a);
        write_f32s(&mut rt, b_ptr, &b);

        let host: f32 = rt
            .invoke_typed("dot_host", (a_ptr as i32, b_ptr as i32, len as i32))
            .unwrap();
        let guest: f32 = rt
            .invoke_typed("dot_guest", (a_ptr as i32, b_ptr as i32, len as i32))
            .unwrap();
        let native = kernels::dot_product_scalar(&a, &b).unwrap();

        let tol = 1e-4 * native.abs().max(1.0);
        assert!(
            (host - guest).abs() <= tol,
            "len={len}: host {host} vs guest {guest}"
        );
        assert!(
            (host - native).abs() <= tol,
            "len={len}: host {host} vs native {native}"
        );
    }
}

#[test]
fn barq_l2_norm_and_cosine_match_native() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module
          (import "barq" "l2_norm_f32" (func $norm (param i32 i32) (result f32)))
          (import "barq" "cosine_similarity_f32" (func $cos (param i32 i32 i32) (result f32)))
          (memory (export "memory") 4)
          (func (export "norm") (param i32 i32) (result f32)
            (call $norm (local.get 0) (local.get 1)))
          (func (export "cos") (param i32 i32 i32) (result f32)
            (call $cos (local.get 0) (local.get 1) (local.get 2))))"#,
    );
    let a = pseudo_random_f32(500, 5);
    let b = pseudo_random_f32(500, 6);
    write_f32s(&mut rt, 0, &a);
    write_f32s(&mut rt, 2048, &b);

    let norm: f32 = rt.invoke_typed("norm", (0i32, 500i32)).unwrap();
    let native_norm = kernels::l2_norm_scalar(&a).unwrap();
    assert!((norm - native_norm).abs() <= 1e-4 * native_norm.max(1.0));

    let cos: f32 = rt.invoke_typed("cos", (0i32, 2048i32, 500i32)).unwrap();
    let native_cos = kernels::cosine_similarity_scalar(&a, &b).unwrap();
    assert!((cos - native_cos).abs() <= 1e-3);
}

#[test]
fn barq_quantize_writes_bit_exact_result_into_guest_memory() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module
          (import "barq" "quantize_i8" (func $q (param i32 i32 i32 f32)))
          (memory (export "memory") 4)
          (func (export "quant") (param i32 i32 i32 f32)
            (call $q (local.get 0) (local.get 1) (local.get 2) (local.get 3))))"#,
    );
    let input = pseudo_random_f32(1000, 7);
    write_f32s(&mut rt, 0, &input);
    rt.invoke_dynamic(
        "quant",
        &[
            WasmValue::I32(0),
            WasmValue::I32(8192),
            WasmValue::I32(1000),
            WasmValue::F32(0.25),
        ],
    )
    .unwrap();
    let out = rt.read_memory(8192, 1000).unwrap();
    let got: Vec<i8> = out.iter().map(|&b| b as i8).collect();
    let expected = kernels::quantize_i8_scalar(&input, 0.25).unwrap();
    assert_eq!(
        got, expected,
        "quantization through the ABI must be bit-exact"
    );
}

#[test]
fn barq_abi_out_of_bounds_traps() {
    let mut rt = runtime();
    load(&mut rt, DOT_MODULE);
    // 4 pages = 256 KiB; ask for a range far past the end.
    let err = rt
        .invoke_dynamic(
            "dot_host",
            &[
                WasmValue::I32(0),
                WasmValue::I32(0),
                WasmValue::I32(10_000_000),
            ],
        )
        .unwrap_err();
    match err {
        BarqError::Trap(msg) => assert!(
            msg.contains("out of bounds") || msg.contains("barq ABI"),
            "trap should describe the bounds error: {msg}"
        ),
        other => panic!("expected Trap, got {other:?}"),
    }
}

#[test]
fn barq_abi_misaligned_pointer_traps() {
    let mut rt = runtime();
    load(&mut rt, DOT_MODULE);
    let err = rt
        .invoke_dynamic(
            "dot_host",
            &[WasmValue::I32(2), WasmValue::I32(0), WasmValue::I32(4)],
        )
        .unwrap_err();
    assert!(matches!(err, BarqError::Trap(_)));
}

#[test]
fn barq_abi_disabled_fails_instantiation() {
    let mut rt = Runtime::new(RuntimeConfig {
        enable_barq_abi: false,
        ..RuntimeConfig::default()
    })
    .unwrap();
    let bytes = wat::parse_str(DOT_MODULE).unwrap();
    rt.load_module(&bytes).unwrap();
    assert!(matches!(rt.instantiate(), Err(BarqError::Instantiation(_))));
}
