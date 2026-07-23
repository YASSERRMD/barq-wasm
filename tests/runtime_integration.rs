//! Integration tests for the Wasmtime-backed runtime.
//!
//! Every fixture is real WAT compiled to a binary module at test time; every
//! assertion checks concrete returned values, not just `Result::is_ok()`.

#![cfg(feature = "native-runtime")]

use barq_wasm::error::BarqError;
use barq_wasm::runtime::{Runtime, RuntimeConfig, WasmValue};
use std::time::Duration;

fn runtime() -> Runtime {
    Runtime::new(RuntimeConfig::default()).expect("runtime construction")
}

fn load(rt: &mut Runtime, wat: &str) {
    let bytes = wat::parse_str(wat).expect("fixture WAT must assemble");
    rt.load_module(&bytes).expect("load_module");
    rt.instantiate().expect("instantiate");
}

// 1. Integer addition
#[test]
fn invokes_integer_addition() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module (func (export "add") (param i32 i32) (result i32)
             local.get 0 local.get 1 i32.add))"#,
    );
    let sum: i32 = rt.invoke_typed("add", (20i32, 22i32)).unwrap();
    assert_eq!(sum, 42);

    let dynamic = rt
        .invoke_dynamic("add", &[WasmValue::I32(-5), WasmValue::I32(7)])
        .unwrap();
    assert_eq!(dynamic, vec![WasmValue::I32(2)]);
}

// 2. Floating-point multiplication
#[test]
fn invokes_float_multiplication() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module (func (export "fmul") (param f64 f64) (result f64)
             local.get 0 local.get 1 f64.mul))"#,
    );
    let product: f64 = rt.invoke_typed("fmul", (1.5f64, 4.0f64)).unwrap();
    assert_eq!(product, 6.0);
}

// 3. Memory write/read through exported functions and host-side access
#[test]
fn reads_and_writes_linear_memory() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module
             (memory (export "memory") 1)
             (func (export "store") (param i32 i32)
               local.get 0 local.get 1 i32.store)
             (func (export "load") (param i32) (result i32)
               local.get 0 i32.load))"#,
    );
    rt.invoke_dynamic("store", &[WasmValue::I32(64), WasmValue::I32(0xBEEF)])
        .unwrap();
    let loaded = rt.invoke_dynamic("load", &[WasmValue::I32(64)]).unwrap();
    assert_eq!(loaded, vec![WasmValue::I32(0xBEEF)]);

    // Host-side access sees the same bytes.
    let bytes = rt.read_memory(64, 4).unwrap();
    assert_eq!(bytes, 0xBEEFi32.to_le_bytes().to_vec());

    rt.write_memory(128, &1234i32.to_le_bytes()).unwrap();
    let via_guest = rt.invoke_dynamic("load", &[WasmValue::I32(128)]).unwrap();
    assert_eq!(via_guest, vec![WasmValue::I32(1234)]);

    // Out-of-bounds host reads are typed errors.
    let oob = rt.read_memory(usize::MAX - 3, 4);
    assert!(matches!(oob, Err(BarqError::MemoryAccess(_))));
}

// 4. Loop execution
#[test]
fn executes_loops() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module (func (export "sum_to") (param i32) (result i64)
             (local i64) (local i32)
             (block $done
               (loop $next
                 local.get 2 local.get 0 i32.ge_s
                 br_if $done
                 local.get 2 i32.const 1 i32.add local.tee 2
                 i64.extend_i32_s
                 local.get 1 i64.add local.set 1
                 br $next))
             local.get 1))"#,
    );
    let sum: i64 = rt.invoke_typed("sum_to", 1000i32).unwrap();
    assert_eq!(sum, 500_500);
}

// 5. Invalid module
#[test]
fn rejects_invalid_module() {
    let mut rt = runtime();
    let garbage = b"not a wasm module at all";
    assert!(matches!(
        rt.validate(garbage),
        Err(BarqError::Validation(_))
    ));
    assert!(matches!(
        rt.load_module(garbage),
        Err(BarqError::Validation(_))
    ));
    // Structurally invalid: body type mismatch.
    let bad = wat::parse_str(r#"(module (func (export "f") (result i32) f32.const 1.5))"#);
    // wat assembles it; wasmtime validation must reject the type error.
    if let Ok(bytes) = bad {
        assert!(matches!(rt.validate(&bytes), Err(BarqError::Validation(_))));
    }
}

// 6. Missing export
#[test]
fn missing_export_is_typed_error() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module (func (export "exists") (result i32) i32.const 1))"#,
    );
    let err = rt.invoke_dynamic("does_not_exist", &[]).unwrap_err();
    match err {
        BarqError::MissingExport { name, .. } => assert_eq!(name, "does_not_exist"),
        other => panic!("expected MissingExport, got {other:?}"),
    }
}

// 7. Function trap
#[test]
fn guest_trap_is_typed_error() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module
             (func (export "boom") (result i32) unreachable)
             (func (export "div") (param i32 i32) (result i32)
               local.get 0 local.get 1 i32.div_s))"#,
    );
    assert!(matches!(
        rt.invoke_dynamic("boom", &[]),
        Err(BarqError::Trap(_))
    ));
    let div_zero = rt
        .invoke_dynamic("div", &[WasmValue::I32(1), WasmValue::I32(0)])
        .unwrap_err();
    match div_zero {
        BarqError::Trap(msg) => assert!(
            msg.to_lowercase().contains("divide"),
            "trap message should mention division: {msg}"
        ),
        other => panic!("expected Trap, got {other:?}"),
    }
}

// 8. Fuel exhaustion
#[test]
fn fuel_exhaustion_is_typed_error() {
    let mut rt = Runtime::new(RuntimeConfig {
        fuel: Some(10_000),
        ..RuntimeConfig::default()
    })
    .unwrap();
    load(
        &mut rt,
        r#"(module (func (export "spin")
             (loop $l br $l)))"#,
    );
    let err = rt.invoke_dynamic("spin", &[]).unwrap_err();
    match err {
        BarqError::FuelExhausted { consumed } => {
            assert!(consumed > 0, "consumed fuel must be reported");
        }
        other => panic!("expected FuelExhausted, got {other:?}"),
    }

    // A cheap call within budget still works and reports consumption.
    let mut rt2 = Runtime::new(RuntimeConfig {
        fuel: Some(10_000),
        ..RuntimeConfig::default()
    })
    .unwrap();
    load(
        &mut rt2,
        r#"(module (func (export "one") (result i32) i32.const 1))"#,
    );
    let one: i32 = rt2.invoke_typed("one", ()).unwrap();
    assert_eq!(one, 1);
    let consumed = rt2.fuel_consumed().expect("fuel accounting enabled");
    assert!(consumed > 0 && consumed < 10_000);
}

// 8b. Wall-clock timeout via epoch interruption
#[test]
fn timeout_interrupts_infinite_loop() {
    let mut rt = Runtime::new(RuntimeConfig {
        timeout: Some(Duration::from_millis(200)),
        ..RuntimeConfig::default()
    })
    .unwrap();
    load(
        &mut rt,
        r#"(module (func (export "spin")
             (loop $l br $l)))"#,
    );
    let start = std::time::Instant::now();
    let err = rt.invoke_dynamic("spin", &[]).unwrap_err();
    assert!(
        matches!(err, BarqError::Timeout { .. }),
        "expected Timeout, got {err:?}"
    );
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "deadline must interrupt promptly"
    );
}

// 8c. Memory growth beyond the configured limit fails
#[test]
fn memory_limit_blocks_growth() {
    let mut rt = Runtime::new(RuntimeConfig {
        max_memory_bytes: Some(2 * 65536), // two pages
        ..RuntimeConfig::default()
    })
    .unwrap();
    load(
        &mut rt,
        r#"(module
             (memory (export "memory") 1)
             (func (export "grow") (param i32) (result i32)
               local.get 0 memory.grow))"#,
    );
    // Growing by one page (to the 2-page limit) succeeds; old size 1 returned.
    let ok = rt.invoke_dynamic("grow", &[WasmValue::I32(1)]).unwrap();
    assert_eq!(ok, vec![WasmValue::I32(1)]);
    // Growing past the limit returns -1 per wasm semantics.
    let blocked = rt.invoke_dynamic("grow", &[WasmValue::I32(1)]).unwrap();
    assert_eq!(blocked, vec![WasmValue::I32(-1)]);
}

// 9. WASI hello world with captured stdout
#[test]
fn wasi_hello_world_writes_stdout() {
    let mut rt = Runtime::new(RuntimeConfig {
        inherit_stdio: false,
        ..RuntimeConfig::default()
    })
    .unwrap();
    load(
        &mut rt,
        r#"(module
             (import "wasi_snapshot_preview1" "fd_write"
               (func $fd_write (param i32 i32 i32 i32) (result i32)))
             (memory (export "memory") 1)
             (data (i32.const 8) "hello wasi\n")
             (func (export "_start")
               (i32.store (i32.const 0) (i32.const 8))   ;; iov_base
               (i32.store (i32.const 4) (i32.const 11))  ;; iov_len
               (call $fd_write
                 (i32.const 1)   ;; stdout
                 (i32.const 0)   ;; *iovs
                 (i32.const 1)   ;; iovs_len
                 (i32.const 20)) ;; *nwritten
               drop))"#,
    );
    rt.invoke_dynamic("_start", &[]).unwrap();
    let (stdout, stderr) = rt.take_wasi_output().expect("captured output");
    assert_eq!(String::from_utf8_lossy(&stdout), "hello wasi\n");
    assert!(stderr.is_empty());
}

// 9b. WASI imports absent when disabled
#[test]
fn wasi_disabled_fails_instantiation() {
    let mut rt = Runtime::new(RuntimeConfig {
        enable_wasi: false,
        ..RuntimeConfig::default()
    })
    .unwrap();
    let bytes = wat::parse_str(
        r#"(module
             (import "wasi_snapshot_preview1" "fd_write"
               (func (param i32 i32 i32 i32) (result i32))))"#,
    )
    .unwrap();
    rt.load_module(&bytes).unwrap();
    assert!(matches!(rt.instantiate(), Err(BarqError::Instantiation(_))));
}

// 10. SIMD-containing module executes (host wasmtime SIMD support)
#[test]
fn executes_simd128_module() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module
             (func (export "sum4") (param f32 f32 f32 f32) (result f32)
               (local $v v128)
               (local.set $v (f32x4.splat (local.get 0)))
               (local.set $v (f32x4.replace_lane 1 (local.get $v) (local.get 1)))
               (local.set $v (f32x4.replace_lane 2 (local.get $v) (local.get 2)))
               (local.set $v (f32x4.replace_lane 3 (local.get $v) (local.get 3)))
               (f32.add
                 (f32.add
                   (f32x4.extract_lane 0 (local.get $v))
                   (f32x4.extract_lane 1 (local.get $v)))
                 (f32.add
                   (f32x4.extract_lane 2 (local.get $v))
                   (f32x4.extract_lane 3 (local.get $v))))))"#,
    );
    let sum: f32 = rt
        .invoke_typed("sum4", (1.0f32, 2.0f32, 3.0f32, 4.0f32))
        .unwrap();
    assert_eq!(sum, 10.0);
}

// Module metadata
#[test]
fn enumerates_exports_and_imports() {
    let mut rt = runtime();
    let bytes = wat::parse_str(
        r#"(module
             (import "env" "host_fn" (func (param i32)))
             (memory (export "memory") 1)
             (func (export "f") (result i32) i32.const 7))"#,
    )
    .unwrap();
    rt.load_module(&bytes).unwrap();
    let info = rt.module_info().unwrap();
    let export_names: Vec<_> = info.exports.iter().map(|e| e.name.as_str()).collect();
    assert!(export_names.contains(&"memory"));
    assert!(export_names.contains(&"f"));
    assert_eq!(info.imports.len(), 1);
    assert_eq!(info.imports[0].module, "env");
    assert_eq!(info.imports[0].name, "host_fn");
}

// Invocation before load/instantiate is a typed error
#[test]
fn invoke_before_load_is_typed_error() {
    let mut rt = runtime();
    assert!(matches!(
        rt.invoke_dynamic("anything", &[]),
        Err(BarqError::ModuleNotLoaded(_))
    ));
    assert!(matches!(
        rt.module_info(),
        Err(BarqError::ModuleNotLoaded(_))
    ));
}

// Wrong argument count/type is a typed error
#[test]
fn argument_mismatch_is_typed_error() {
    let mut rt = runtime();
    load(
        &mut rt,
        r#"(module (func (export "add") (param i32 i32) (result i32)
             local.get 0 local.get 1 i32.add))"#,
    );
    assert!(matches!(
        rt.invoke_dynamic("add", &[WasmValue::I32(1)]),
        Err(BarqError::InvalidArgument(_))
    ));
    assert!(matches!(
        rt.invoke_dynamic("add", &[WasmValue::I32(1), WasmValue::F64(2.0)]),
        Err(BarqError::InvalidArgument(_))
    ));
}
