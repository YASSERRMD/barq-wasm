//! Barq-WASM CLI: validate, inspect, run, and benchmark WebAssembly modules.

use barq_wasm::runtime::{Runtime, RuntimeConfig, WasmValue};
use clap::{Arg, ArgAction, Command};
use std::process::ExitCode;
use std::time::{Duration, Instant};

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cli() -> Command {
    let module_arg = Arg::new("module")
        .required(true)
        .value_name("MODULE")
        .help("Path to a .wasm (or .wat) module");
    let invoke_args = [
        Arg::new("invoke")
            .long("invoke")
            .required(true)
            .value_name("EXPORT")
            .help("Name of the exported function to call"),
        Arg::new("arg-i32")
            .long("arg-i32")
            .action(ArgAction::Append)
            .allow_hyphen_values(true)
            .value_name("I32"),
        Arg::new("arg-i64")
            .long("arg-i64")
            .action(ArgAction::Append)
            .allow_hyphen_values(true)
            .value_name("I64"),
        Arg::new("arg-f32")
            .long("arg-f32")
            .action(ArgAction::Append)
            .allow_hyphen_values(true)
            .value_name("F32"),
        Arg::new("arg-f64")
            .long("arg-f64")
            .action(ArgAction::Append)
            .allow_hyphen_values(true)
            .value_name("F64"),
        Arg::new("fuel")
            .long("fuel")
            .value_name("UNITS")
            .help("Deterministic instruction budget; trap when exhausted"),
        Arg::new("timeout-ms")
            .long("timeout-ms")
            .value_name("MS")
            .help("Wall-clock deadline per invocation, in milliseconds"),
        Arg::new("max-memory")
            .long("max-memory")
            .value_name("BYTES")
            .help("Upper bound for guest linear memory"),
        Arg::new("no-wasi")
            .long("no-wasi")
            .action(ArgAction::SetTrue)
            .help("Do not link WASI imports"),
    ];

    Command::new("barq-wasm")
        .about("Barq-WASM: a WebAssembly runtime (Wasmtime-backed)")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("validate")
                .about("Validate a WebAssembly module")
                .arg(module_arg.clone()),
        )
        .subcommand(
            Command::new("inspect")
                .about("List a module's imports and exports")
                .arg(module_arg.clone()),
        )
        .subcommand(
            Command::new("run")
                .about("Instantiate a module and invoke an export")
                .arg(module_arg.clone())
                .args(invoke_args.clone()),
        )
        .subcommand(
            Command::new("benchmark")
                .about("Repeatedly invoke an export and report wall-clock statistics")
                .arg(module_arg)
                .args(invoke_args)
                .arg(
                    Arg::new("iterations")
                        .long("iterations")
                        .default_value("100")
                        .value_name("N"),
                )
                .arg(
                    Arg::new("warmup")
                        .long("warmup")
                        .default_value("10")
                        .value_name("N"),
                ),
        )
}

fn real_main() -> Result<(), String> {
    let matches = cli().get_matches();
    match matches.subcommand() {
        Some(("validate", sub)) => {
            let bytes = read_module(sub)?;
            let runtime = Runtime::new(RuntimeConfig::default()).map_err(stringify)?;
            runtime.validate(&bytes).map_err(stringify)?;
            println!("valid");
            Ok(())
        }
        Some(("inspect", sub)) => {
            let bytes = read_module(sub)?;
            let mut runtime = Runtime::new(RuntimeConfig::default()).map_err(stringify)?;
            runtime.load_module(&bytes).map_err(stringify)?;
            let info = runtime.module_info().map_err(stringify)?;
            println!("imports:");
            for i in &info.imports {
                println!("  {}::{} ({})", i.module, i.name, i.kind);
            }
            println!("exports:");
            for e in &info.exports {
                println!("  {} ({})", e.name, e.kind);
            }
            Ok(())
        }
        Some(("run", sub)) => {
            let (mut runtime, export, args) = prepare(sub)?;
            let results = runtime.invoke_dynamic(&export, &args).map_err(stringify)?;
            match results.len() {
                0 => println!("{export} returned"),
                _ => println!(
                    "{}",
                    results
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                ),
            }
            if let Some(consumed) = runtime.fuel_consumed() {
                eprintln!("fuel consumed: {consumed}");
            }
            Ok(())
        }
        Some(("benchmark", sub)) => {
            let (mut runtime, export, args) = prepare(sub)?;
            let iterations: usize = parse_num(sub, "iterations")?;
            let warmup: usize = parse_num(sub, "warmup")?;

            // Correctness/consumption first: the result must be produced and
            // shown so a broken function can never publish a timing.
            let reference = runtime.invoke_dynamic(&export, &args).map_err(stringify)?;

            for _ in 0..warmup {
                runtime.invoke_dynamic(&export, &args).map_err(stringify)?;
            }
            let mut samples_ns: Vec<u128> = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                let out = runtime.invoke_dynamic(&export, &args).map_err(stringify)?;
                samples_ns.push(start.elapsed().as_nanos());
                if out != reference {
                    return Err(format!(
                        "non-deterministic result during benchmark of '{export}'"
                    ));
                }
            }
            samples_ns.sort_unstable();
            let median = samples_ns[samples_ns.len() / 2];
            let p95 = samples_ns[(samples_ns.len() * 95 / 100).min(samples_ns.len() - 1)];
            println!(
                "result: {}",
                reference
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!("iterations: {iterations} (after {warmup} warmup)");
            println!("min: {} ns", samples_ns[0]);
            println!("median: {median} ns");
            println!("p95: {p95} ns");
            println!("max: {} ns", samples_ns[samples_ns.len() - 1]);
            Ok(())
        }
        _ => unreachable!("subcommand required"),
    }
}

fn read_module(sub: &clap::ArgMatches) -> Result<Vec<u8>, String> {
    let path = sub.get_one::<String>("module").expect("required arg");
    std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))
}

fn parse_num<T: std::str::FromStr>(sub: &clap::ArgMatches, name: &str) -> Result<T, String> {
    sub.get_one::<String>(name)
        .expect("has default")
        .parse::<T>()
        .map_err(|_| format!("--{name} must be a number"))
}

fn stringify(e: barq_wasm::BarqError) -> String {
    e.to_string()
}

/// Collect typed `--arg-*` values in their original command-line order.
fn collect_args(sub: &clap::ArgMatches) -> Result<Vec<WasmValue>, String> {
    let mut ordered: Vec<(usize, WasmValue)> = Vec::new();
    macro_rules! gather {
        ($flag:literal, $ty:ty, $variant:ident) => {
            if let (Some(values), Some(indices)) =
                (sub.get_many::<String>($flag), sub.indices_of($flag))
            {
                for (raw, idx) in values.zip(indices) {
                    let parsed: $ty = raw
                        .parse()
                        .map_err(|_| format!("--{}: '{raw}' is not a valid value", $flag))?;
                    ordered.push((idx, WasmValue::$variant(parsed)));
                }
            }
        };
    }
    gather!("arg-i32", i32, I32);
    gather!("arg-i64", i64, I64);
    gather!("arg-f32", f32, F32);
    gather!("arg-f64", f64, F64);
    ordered.sort_by_key(|(idx, _)| *idx);
    Ok(ordered.into_iter().map(|(_, v)| v).collect())
}

fn prepare(sub: &clap::ArgMatches) -> Result<(Runtime, String, Vec<WasmValue>), String> {
    let bytes = read_module(sub)?;
    let export = sub
        .get_one::<String>("invoke")
        .expect("required arg")
        .clone();
    let args = collect_args(sub)?;

    let mut config = RuntimeConfig {
        enable_wasi: !sub.get_flag("no-wasi"),
        ..RuntimeConfig::default()
    };
    if let Some(fuel) = sub.get_one::<String>("fuel") {
        config.fuel = Some(fuel.parse().map_err(|_| "--fuel must be a number")?);
    }
    if let Some(ms) = sub.get_one::<String>("timeout-ms") {
        let ms: u64 = ms.parse().map_err(|_| "--timeout-ms must be a number")?;
        config.timeout = Some(Duration::from_millis(ms));
    }
    if let Some(bytes) = sub.get_one::<String>("max-memory") {
        config.max_memory_bytes = Some(bytes.parse().map_err(|_| "--max-memory must be a number")?);
    }

    let mut runtime = Runtime::new(config).map_err(stringify)?;
    runtime.load_module(&bytes).map_err(stringify)?;
    runtime.instantiate().map_err(stringify)?;
    Ok((runtime, export, args))
}
