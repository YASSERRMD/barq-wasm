//! Barq host-kernel ABI: real, opt-in acceleration.
//!
//! A guest module that imports functions from the `barq` module gets them
//! resolved to the crate's verified native kernels (AVX2/FMA, NEON, or
//! scalar — whatever the host CPU supports, via the same dispatch used
//! everywhere else). This is host-kernel substitution: the guest opts in by
//! importing the ABI; nothing is silently rewritten.
//!
//! ABI (all pointers are byte offsets into the guest's exported "memory";
//! lengths are element counts):
//!
//! - `barq.dot_product_f32(a_ptr: i32, b_ptr: i32, len: i32) -> f32`
//! - `barq.l2_norm_f32(ptr: i32, len: i32) -> f32`
//! - `barq.cosine_similarity_f32(a_ptr: i32, b_ptr: i32, len: i32) -> f32`
//! - `barq.quantize_i8(src_ptr: i32, dst_ptr: i32, len: i32, scale: f32) -> ()`
//!
//! Out-of-bounds ranges and misaligned f32 pointers trap; traps surface as
//! typed `BarqError::Trap` values on the invoking side.

use super::RuntimeState;
use crate::error::{BarqError, BarqResult};
use crate::kernels;
use wasmtime::{Caller, Linker};

fn guest_memory(
    caller: &mut Caller<'_, RuntimeState>,
) -> Result<wasmtime::Memory, wasmtime::Error> {
    caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .ok_or_else(|| {
            wasmtime::Error::msg(
                "barq ABI requires the guest to export its linear memory as 'memory'",
            )
        })
}

/// Read a f32 slice out of guest memory (copies; the kernel may run SIMD
/// over it without aliasing guest memory).
fn read_f32s(
    caller: &mut Caller<'_, RuntimeState>,
    ptr: u32,
    len: u32,
) -> Result<Vec<f32>, wasmtime::Error> {
    let memory = guest_memory(caller)?;
    let start = ptr as usize;
    let bytes_len = (len as usize)
        .checked_mul(4)
        .ok_or_else(|| wasmtime::Error::msg("barq ABI: length overflow"))?;
    if !ptr.is_multiple_of(4) {
        return Err(wasmtime::Error::msg(
            "barq ABI: f32 pointer must be 4-byte aligned",
        ));
    }
    let data = memory.data(caller);
    let end = start
        .checked_add(bytes_len)
        .filter(|&e| e <= data.len())
        .ok_or_else(|| wasmtime::Error::msg("barq ABI: range out of bounds"))?;
    let mut out = vec![0f32; len as usize];
    for (i, chunk) in data[start..end].chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

fn kernel_err(e: BarqError) -> wasmtime::Error {
    wasmtime::Error::msg(e.to_string())
}

/// Register the barq host kernels on a linker.
pub fn add_to_linker(linker: &mut Linker<RuntimeState>) -> BarqResult<()> {
    linker
        .func_wrap(
            "barq",
            "dot_product_f32",
            |mut caller: Caller<'_, RuntimeState>, a: u32, b: u32, len: u32| {
                let va = read_f32s(&mut caller, a, len)?;
                let vb = read_f32s(&mut caller, b, len)?;
                kernels::dot_product(&va, &vb).map_err(kernel_err)
            },
        )
        .map_err(|e| BarqError::RuntimeNotInitialized(e.to_string()))?;

    linker
        .func_wrap(
            "barq",
            "l2_norm_f32",
            |mut caller: Caller<'_, RuntimeState>, ptr: u32, len: u32| {
                let v = read_f32s(&mut caller, ptr, len)?;
                kernels::l2_norm(&v).map_err(kernel_err)
            },
        )
        .map_err(|e| BarqError::RuntimeNotInitialized(e.to_string()))?;

    linker
        .func_wrap(
            "barq",
            "cosine_similarity_f32",
            |mut caller: Caller<'_, RuntimeState>, a: u32, b: u32, len: u32| {
                let va = read_f32s(&mut caller, a, len)?;
                let vb = read_f32s(&mut caller, b, len)?;
                kernels::cosine_similarity(&va, &vb).map_err(kernel_err)
            },
        )
        .map_err(|e| BarqError::RuntimeNotInitialized(e.to_string()))?;

    linker
        .func_wrap(
            "barq",
            "quantize_i8",
            |mut caller: Caller<'_, RuntimeState>, src: u32, dst: u32, len: u32, scale: f32| {
                let input = read_f32s(&mut caller, src, len)?;
                let quantized = kernels::quantize_i8(&input, scale).map_err(kernel_err)?;
                let memory = guest_memory(&mut caller)?;
                let bytes: Vec<u8> = quantized.iter().map(|&q| q as u8).collect();
                memory
                    .write(&mut caller, dst as usize, &bytes)
                    .map_err(|e| wasmtime::Error::msg(format!("barq ABI: {e}")))?;
                Ok(())
            },
        )
        .map_err(|e| BarqError::RuntimeNotInitialized(e.to_string()))?;

    Ok(())
}
