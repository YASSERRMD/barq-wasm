#![cfg(all(not(target_arch = "wasm32"), feature = "analyzer"))]

//! Fuzz smoke tests: randomized and mutated inputs must never panic the
//! analyzer, the validator, or the kernels — typed errors are fine, crashes
//! are not. (Deeper coverage-guided fuzzing needs nightly cargo-fuzz; this
//! smoke layer runs everywhere, including CI.)

use barq_wasm::analyzer::analyze_module;
use barq_wasm::kernels;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

#[test]
fn analyzer_survives_random_bytes() {
    let mut rng = Lcg(0x5EED);
    for len in [0usize, 1, 7, 64, 512, 4096] {
        for _ in 0..50 {
            let bytes: Vec<u8> = (0..len).map(|_| (rng.next() >> 32) as u8).collect();
            let _ = analyze_module(&bytes); // must not panic
        }
    }
}

#[test]
fn analyzer_survives_mutated_valid_modules() {
    let base = wat::parse_str(
        r#"(module (memory 1)
          (func (export "dot") (param $a i32) (param $b i32) (param $n i32) (result f32)
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
            (local.get $sum)))"#,
    )
    .unwrap();
    let mut rng = Lcg(0xBEEF);
    for _ in 0..500 {
        let mut mutated = base.clone();
        let flips = 1 + (rng.next() as usize % 8);
        for _ in 0..flips {
            let idx = rng.next() as usize % mutated.len();
            mutated[idx] ^= (rng.next() >> 24) as u8;
        }
        let _ = analyze_module(&mutated); // must not panic
    }
}

#[cfg(feature = "native-runtime")]
#[test]
fn runtime_validation_survives_mutated_modules() {
    use barq_wasm::runtime::{Runtime, RuntimeConfig};
    let base = wat::parse_str(r#"(module (func (export "f") (result i32) i32.const 1))"#).unwrap();
    let rt = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut rng = Lcg(0xCAFE);
    for _ in 0..300 {
        let mut mutated = base.clone();
        let idx = rng.next() as usize % mutated.len();
        mutated[idx] ^= (rng.next() >> 24) as u8;
        let _ = rt.validate(&mutated); // must not panic
    }
}

#[test]
fn kernels_survive_edge_lengths_and_scales() {
    let mut rng = Lcg(0xF00D);
    for _ in 0..200 {
        let len = rng.next() as usize % 100;
        let data: Vec<f32> = (0..len)
            .map(|_| f32::from_bits((rng.next() >> 32) as u32))
            .collect();
        // Any bit pattern (NaNs, infs, denormals) must be handled without
        // panicking, for every scale class.
        for scale in [1.0f32, f32::MIN_POSITIVE, 1e30] {
            let _ = kernels::quantize_i8(&data, scale);
        }
        let _ = kernels::l2_norm(&data);
        let _ = kernels::dot_product(&data, &data);
    }
}
