//! Real WebAssembly runtime backed by Wasmtime.
//!
//! Supports module validation, instantiation, typed and dynamic invocation,
//! linear memory access, WASI preview1, fuel limits, epoch-based wall-clock
//! deadlines, and memory limits. Traps are mapped to typed [`BarqError`]
//! variants.

mod config;
mod value;

pub use config::RuntimeConfig;
pub use value::WasmValue;

use crate::error::{BarqError, BarqResult};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;
use wasmtime::{
    Engine, Instance, Linker, Module, Store, StoreLimits, StoreLimitsBuilder, Trap, Val,
};
use wasmtime_wasi::pipe::MemoryOutputPipe;
use wasmtime_wasi::preview1::WasiP1Ctx;
use wasmtime_wasi::WasiCtxBuilder;

/// Per-store host state: WASI context and resource limits.
pub struct RuntimeState {
    wasi: WasiP1Ctx,
    limits: StoreLimits,
}

/// Captured WASI output pipes (when stdio is not inherited).
struct CapturedOutput {
    stdout: MemoryOutputPipe,
    stderr: MemoryOutputPipe,
}

/// A Wasmtime-backed WebAssembly runtime.
pub struct Runtime {
    engine: Engine,
    module: Option<Module>,
    store: Store<RuntimeState>,
    linker: Linker<RuntimeState>,
    instance: Option<Instance>,
    config: RuntimeConfig,
    captured: Option<CapturedOutput>,
}

/// Description of one module export.
#[derive(Debug, Clone)]
pub struct ExportInfo {
    pub name: String,
    pub kind: String,
    pub type_signature: String,
}

/// Description of one module import.
#[derive(Debug, Clone)]
pub struct ImportInfo {
    pub module: String,
    pub name: String,
    pub kind: String,
}

/// Metadata about a loaded module.
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub exports: Vec<ExportInfo>,
    pub imports: Vec<ImportInfo>,
}

impl Runtime {
    /// Create a runtime with the given configuration.
    pub fn new(config: RuntimeConfig) -> BarqResult<Self> {
        let mut wasmtime_config = wasmtime::Config::new();
        wasmtime_config.consume_fuel(config.fuel.is_some());
        wasmtime_config.epoch_interruption(config.timeout.is_some());

        let engine = Engine::new(&wasmtime_config)
            .map_err(|e| BarqError::RuntimeNotInitialized(e.to_string()))?;

        let mut builder = WasiCtxBuilder::new();
        builder.args(&config.wasi_args);
        let mut captured = None;
        if config.inherit_stdio {
            builder.inherit_stdout().inherit_stderr();
        } else {
            let stdout = MemoryOutputPipe::new(1 << 20);
            let stderr = MemoryOutputPipe::new(1 << 20);
            builder.stdout(stdout.clone()).stderr(stderr.clone());
            captured = Some(CapturedOutput { stdout, stderr });
        }
        let wasi = builder.build_p1();

        let mut limits_builder = StoreLimitsBuilder::new();
        if let Some(bytes) = config.max_memory_bytes {
            limits_builder = limits_builder.memory_size(bytes);
        }
        let state = RuntimeState {
            wasi,
            limits: limits_builder.build(),
        };

        let mut store = Store::new(&engine, state);
        store.limiter(|state| &mut state.limits);

        let mut linker: Linker<RuntimeState> = Linker::new(&engine);
        if config.enable_wasi {
            wasmtime_wasi::preview1::add_to_linker_sync(&mut linker, |state| &mut state.wasi)
                .map_err(|e| BarqError::RuntimeNotInitialized(e.to_string()))?;
        }

        Ok(Self {
            engine,
            module: None,
            store,
            linker,
            instance: None,
            config,
            captured,
        })
    }

    /// Validate a byte stream as a WebAssembly module without loading it.
    pub fn validate(&self, bytes: &[u8]) -> BarqResult<()> {
        Module::validate(&self.engine, bytes).map_err(|e| BarqError::Validation(e.to_string()))
    }

    /// Compile and hold a module. Accepts binary `.wasm` (and `.wat` text).
    pub fn load_module(&mut self, bytes: &[u8]) -> BarqResult<()> {
        let module =
            Module::new(&self.engine, bytes).map_err(|e| BarqError::Validation(e.to_string()))?;
        self.module = Some(module);
        self.instance = None;
        Ok(())
    }

    /// Instantiate the loaded module, resolving imports through the linker.
    pub fn instantiate(&mut self) -> BarqResult<()> {
        let module = self.module.as_ref().ok_or_else(|| {
            BarqError::ModuleNotLoaded("call load_module before instantiate".to_string())
        })?;
        // Instantiation can run a start function; meter it like an invocation.
        if let Some(fuel) = self.config.fuel {
            self.store
                .set_fuel(fuel)
                .map_err(|e| BarqError::RuntimeNotInitialized(e.to_string()))?;
        }
        let instance = self
            .linker
            .instantiate(&mut self.store, module)
            .map_err(|e| match map_wasmtime_error(&e, &mut self.store, self.config.fuel) {
                Some(mapped) => mapped,
                None => BarqError::Instantiation(format!("{e:#}")),
            })?;
        self.instance = Some(instance);
        Ok(())
    }

    /// Invoke an exported function with statically-typed params/results.
    pub fn invoke_typed<P, R>(&mut self, name: &str, params: P) -> BarqResult<R>
    where
        P: wasmtime::WasmParams,
        R: wasmtime::WasmResults,
    {
        let instance = self.instance.ok_or_else(|| {
            BarqError::ModuleNotLoaded("call instantiate before invoking".to_string())
        })?;
        let func = instance
            .get_typed_func::<P, R>(&mut self.store, name)
            .map_err(|e| BarqError::MissingExport {
                name: name.to_string(),
                reason: format!("{e:#}"),
            })?;
        self.prepare_invocation()?;
        let deadline = DeadlineGuard::arm(&self.engine, &mut self.store, self.config.timeout);
        let result = func.call(&mut self.store, params);
        let timed_out = deadline.map(|d| d.finish()).unwrap_or(false);
        result.map_err(|e| self.map_call_error(e, timed_out))
    }

    /// Invoke an exported function with dynamically-typed values.
    pub fn invoke_dynamic(&mut self, name: &str, args: &[WasmValue]) -> BarqResult<Vec<WasmValue>> {
        let instance = self.instance.ok_or_else(|| {
            BarqError::ModuleNotLoaded("call instantiate before invoking".to_string())
        })?;
        let func = instance.get_func(&mut self.store, name).ok_or_else(|| {
            BarqError::MissingExport {
                name: name.to_string(),
                reason: "no function export with this name".to_string(),
            }
        })?;
        let ty = func.ty(&self.store);
        let expected: Vec<_> = ty.params().collect();
        if expected.len() != args.len() {
            return Err(BarqError::InvalidArgument(format!(
                "'{name}' expects {} argument(s), got {}",
                expected.len(),
                args.len()
            )));
        }
        let vals: Vec<Val> = args
            .iter()
            .zip(expected.iter())
            .enumerate()
            .map(|(i, (arg, want))| {
                let matches_ty = matches!(
                    (arg, want),
                    (WasmValue::I32(_), wasmtime::ValType::I32)
                        | (WasmValue::I64(_), wasmtime::ValType::I64)
                        | (WasmValue::F32(_), wasmtime::ValType::F32)
                        | (WasmValue::F64(_), wasmtime::ValType::F64)
                );
                if matches_ty {
                    Ok(arg.to_val())
                } else {
                    Err(BarqError::InvalidArgument(format!(
                        "argument {i} of '{name}': expected {want}, got {}",
                        arg.type_name()
                    )))
                }
            })
            .collect::<BarqResult<_>>()?;
        let mut results = vec![Val::I32(0); ty.results().len()];
        self.prepare_invocation()?;
        let deadline = DeadlineGuard::arm(&self.engine, &mut self.store, self.config.timeout);
        let outcome = func.call(&mut self.store, &vals, &mut results);
        let timed_out = deadline.map(|d| d.finish()).unwrap_or(false);
        outcome.map_err(|e| self.map_call_error(e, timed_out))?;
        results.iter().map(WasmValue::from_val).collect()
    }

    /// Fuel consumed by the most recent invocation, when fuel is enabled.
    pub fn fuel_consumed(&self) -> Option<u64> {
        let initial = self.config.fuel?;
        let remaining = self.store.get_fuel().ok()?;
        Some(initial.saturating_sub(remaining))
    }

    /// The instance's exported linear memory (named "memory").
    pub fn get_memory(&mut self) -> BarqResult<wasmtime::Memory> {
        let instance = self.instance.ok_or_else(|| {
            BarqError::ModuleNotLoaded("call instantiate before accessing memory".to_string())
        })?;
        instance
            .get_memory(&mut self.store, "memory")
            .ok_or_else(|| BarqError::MemoryAccess("module exports no memory".to_string()))
    }

    /// Read `len` bytes at `offset` from the exported linear memory.
    pub fn read_memory(&mut self, offset: usize, len: usize) -> BarqResult<Vec<u8>> {
        let memory = self.get_memory()?;
        let mut buf = vec![0u8; len];
        memory
            .read(&self.store, offset, &mut buf)
            .map_err(|e| BarqError::MemoryAccess(e.to_string()))?;
        Ok(buf)
    }

    /// Write bytes at `offset` into the exported linear memory.
    pub fn write_memory(&mut self, offset: usize, data: &[u8]) -> BarqResult<()> {
        let memory = self.get_memory()?;
        memory
            .write(&mut self.store, offset, data)
            .map_err(|e| BarqError::MemoryAccess(e.to_string()))
    }

    /// Enumerate exports and imports of the loaded module.
    pub fn module_info(&self) -> BarqResult<ModuleInfo> {
        let module = self.module.as_ref().ok_or_else(|| {
            BarqError::ModuleNotLoaded("call load_module before inspecting".to_string())
        })?;
        let exports = module
            .exports()
            .map(|e| ExportInfo {
                name: e.name().to_string(),
                kind: extern_kind(&e.ty()),
                type_signature: format!("{:?}", e.ty()),
            })
            .collect();
        let imports = module
            .imports()
            .map(|i| ImportInfo {
                module: i.module().to_string(),
                name: i.name().to_string(),
                kind: extern_kind(&i.ty()),
            })
            .collect();
        Ok(ModuleInfo { exports, imports })
    }

    /// Captured WASI stdout/stderr contents (only when `inherit_stdio` is false).
    pub fn take_wasi_output(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        self.captured
            .as_ref()
            .map(|c| (c.stdout.contents().to_vec(), c.stderr.contents().to_vec()))
    }

    fn prepare_invocation(&mut self) -> BarqResult<()> {
        if let Some(fuel) = self.config.fuel {
            self.store
                .set_fuel(fuel)
                .map_err(|e| BarqError::RuntimeNotInitialized(e.to_string()))?;
        }
        Ok(())
    }

    fn map_call_error(&mut self, error: anyhow::Error, timed_out: bool) -> BarqError {
        match map_wasmtime_error(&error, &mut self.store, self.config.fuel) {
            Some(BarqError::Trap(msg)) if timed_out => {
                // Epoch interruption surfaces as an Interrupt trap.
                let millis = self.config.timeout.map(|t| t.as_millis() as u64).unwrap_or(0);
                if msg.contains("interrupt") {
                    BarqError::Timeout { millis }
                } else {
                    BarqError::Trap(msg)
                }
            }
            Some(mapped) => mapped,
            None => BarqError::Trap(format!("{error:#}")),
        }
    }
}

fn extern_kind(ty: &wasmtime::ExternType) -> String {
    match ty {
        wasmtime::ExternType::Func(f) => format!("func {f}"),
        wasmtime::ExternType::Memory(_) => "memory".to_string(),
        wasmtime::ExternType::Global(_) => "global".to_string(),
        wasmtime::ExternType::Table(_) => "table".to_string(),
    }
}

fn map_wasmtime_error(
    error: &anyhow::Error,
    store: &mut Store<RuntimeState>,
    fuel: Option<u64>,
) -> Option<BarqError> {
    let trap = error.downcast_ref::<Trap>()?;
    match trap {
        Trap::OutOfFuel => {
            let consumed = fuel
                .map(|f| f.saturating_sub(store.get_fuel().unwrap_or(0)))
                .unwrap_or(0);
            Some(BarqError::FuelExhausted { consumed })
        }
        Trap::Interrupt => Some(BarqError::Trap("interrupt (epoch deadline)".to_string())),
        other => Some(BarqError::Trap(format!("{other:?}: {error:#}"))),
    }
}

/// Arms a wall-clock deadline for one invocation using epoch interruption.
///
/// A watchdog thread waits on a channel with a timeout; if the timeout fires
/// first, it increments the engine epoch, which traps guest execution. The
/// guard cancels the watchdog when the invocation completes.
struct DeadlineGuard {
    cancel: mpsc::Sender<()>,
    handle: std::thread::JoinHandle<()>,
    fired: Arc<AtomicBool>,
}

impl DeadlineGuard {
    fn arm(
        engine: &Engine,
        store: &mut Store<RuntimeState>,
        timeout: Option<Duration>,
    ) -> Option<Self> {
        let timeout = timeout?;
        store.set_epoch_deadline(1);
        let engine = engine.clone();
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = fired.clone();
        let (cancel, rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            if rx.recv_timeout(timeout).is_err() {
                fired_clone.store(true, Ordering::SeqCst);
                engine.increment_epoch();
            }
        });
        Some(Self {
            cancel,
            handle,
            fired,
        })
    }

    /// Cancel the watchdog and report whether the deadline fired.
    fn finish(self) -> bool {
        let _ = self.cancel.send(());
        let _ = self.handle.join();
        self.fired.load(Ordering::SeqCst)
    }
}
