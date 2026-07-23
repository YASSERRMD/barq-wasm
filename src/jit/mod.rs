//! Real Cranelift JIT for narrowly defined kernel signatures.
//!
//! This module compiles complete functions with cranelift-jit, resolves real
//! callable pointers, executes them through typed wrappers, and is
//! differentially tested against the scalar references. It supports exactly
//! the signatures listed here — nothing arbitrary, no user-controlled code,
//! no isolated opcode fragments.
//!
//! Safety model:
//! - IR is constructed entirely by this module; no external input reaches
//!   the code generator.
//! - Executable memory is owned by the wrapper struct and freed on drop;
//!   calls borrow the wrapper, so the pointer cannot outlive the memory.
//! - Pointer/length arguments are validated in safe Rust before the call.

use crate::error::{BarqError, BarqResult};
use cranelift_codegen::ir::{condcodes::IntCC, types, AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

fn spec_err(e: impl std::fmt::Display) -> BarqError {
    BarqError::SpecializationUnavailable(format!("cranelift JIT: {e}"))
}

/// Build a JIT module for the host ISA with non-PIC settings (PLT-based PIC
/// is x86-64-only in cranelift-jit; plain JIT code needs neither).
fn new_jit_module() -> BarqResult<JITModule> {
    let mut flags = settings::builder();
    flags
        .set("use_colocated_libcalls", "false")
        .map_err(spec_err)?;
    flags.set("is_pic", "false").map_err(spec_err)?;
    let isa = cranelift_native::builder()
        .map_err(spec_err)?
        .finish(settings::Flags::new(flags))
        .map_err(spec_err)?;
    let builder = JITBuilder::with_isa(isa, default_libcall_names());
    Ok(JITModule::new(builder))
}

/// A JIT-compiled f32 dot product: `fn(*const f32, *const f32, i64) -> f32`.
pub struct JitDotProduct {
    module: Option<JITModule>,
    func: unsafe extern "C" fn(*const f32, *const f32, i64) -> f32,
}

impl JitDotProduct {
    /// Build the complete function in Cranelift IR, compile it, and resolve
    /// the callable pointer.
    pub fn compile() -> BarqResult<Self> {
        let mut module = new_jit_module()?;
        let mut ctx = module.make_context();
        let mut fb_ctx = FunctionBuilderContext::new();

        let ptr_ty = module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // a
        ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // b
        ctx.func.signature.params.push(AbiParam::new(types::I64)); // len
        ctx.func.signature.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function(
                "barq_jit_dot_product_f32",
                Linkage::Export,
                &ctx.func.signature,
            )
            .map_err(spec_err)?;

        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);
            let (pa, pb, len) = {
                let p = b.block_params(entry);
                (p[0], p[1], p[2])
            };

            // Loop header carries (index, accumulator) as block params.
            let header = b.create_block();
            b.append_block_param(header, types::I64);
            b.append_block_param(header, types::F32);
            let body = b.create_block();
            let exit = b.create_block();

            let zero_i = b.ins().iconst(types::I64, 0);
            let zero_f = b.ins().f32const(0.0);
            b.ins().jump(header, &[zero_i.into(), zero_f.into()]);

            b.switch_to_block(header);
            let i = b.block_params(header)[0];
            let acc = b.block_params(header)[1];
            let done = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, i, len);
            b.ins().brif(done, exit, &[], body, &[]);

            b.switch_to_block(body);
            b.seal_block(body);
            let four = b.ins().iconst(types::I64, 4);
            let off = b.ins().imul(i, four);
            let addr_a = b.ins().iadd(pa, off);
            let addr_b = b.ins().iadd(pb, off);
            let va = b.ins().load(types::F32, MemFlags::trusted(), addr_a, 0);
            let vb = b.ins().load(types::F32, MemFlags::trusted(), addr_b, 0);
            let prod = b.ins().fmul(va, vb);
            let acc_next = b.ins().fadd(acc, prod);
            let one = b.ins().iconst(types::I64, 1);
            let i_next = b.ins().iadd(i, one);
            b.ins().jump(header, &[i_next.into(), acc_next.into()]);
            b.seal_block(header);

            b.switch_to_block(exit);
            b.seal_block(exit);
            b.ins().return_(&[acc]);
            b.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(spec_err)?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(spec_err)?;

        let code = module.get_finalized_function(func_id);
        // SAFETY: the signature transmuted to matches the IR signature built
        // above exactly (ptr, ptr, i64) -> f32 with the C calling convention.
        let func = unsafe {
            std::mem::transmute::<*const u8, unsafe extern "C" fn(*const f32, *const f32, i64) -> f32>(
                code,
            )
        };
        Ok(Self {
            module: Some(module),
            func,
        })
    }

    /// Execute the compiled function over two equal-length slices.
    pub fn call(&self, a: &[f32], b: &[f32]) -> BarqResult<f32> {
        if a.len() != b.len() {
            return Err(BarqError::InvalidArgument(format!(
                "slice lengths differ: {} vs {}",
                a.len(),
                b.len()
            )));
        }
        // SAFETY: pointers come from live slices; the compiled loop reads
        // exactly `len` f32 elements from each.
        Ok(unsafe { (self.func)(a.as_ptr(), b.as_ptr(), a.len() as i64) })
    }
}

impl Drop for JitDotProduct {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: `func` is only callable through &self; once we are in
            // drop there are no outstanding borrows, so no pointer into the
            // JIT memory can be used afterwards.
            unsafe { module.free_memory() };
        }
    }
}

/// A JIT-compiled integer add: `fn(i64, i64) -> i64`. Smoke-test kernel
/// proving end-to-end compile + execute on the host.
pub struct JitAddI64 {
    module: Option<JITModule>,
    func: unsafe extern "C" fn(i64, i64) -> i64,
}

impl JitAddI64 {
    pub fn compile() -> BarqResult<Self> {
        let mut module = new_jit_module()?;
        let mut ctx = module.make_context();
        let mut fb_ctx = FunctionBuilderContext::new();

        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let func_id = module
            .declare_function("barq_jit_add_i64", Linkage::Export, &ctx.func.signature)
            .map_err(spec_err)?;
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);
            let (x, y) = {
                let p = b.block_params(entry);
                (p[0], p[1])
            };
            let sum = b.ins().iadd(x, y);
            b.ins().return_(&[sum]);
            b.finalize();
        }
        module
            .define_function(func_id, &mut ctx)
            .map_err(spec_err)?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(spec_err)?;
        let code = module.get_finalized_function(func_id);
        // SAFETY: signature matches the IR built above.
        let func = unsafe {
            std::mem::transmute::<*const u8, unsafe extern "C" fn(i64, i64) -> i64>(code)
        };
        Ok(Self {
            module: Some(module),
            func,
        })
    }

    pub fn call(&self, a: i64, b: i64) -> i64 {
        // SAFETY: pure arithmetic on register arguments.
        unsafe { (self.func)(a, b) }
    }
}

impl Drop for JitAddI64 {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: see JitDotProduct::drop.
            unsafe { module.free_memory() };
        }
    }
}
