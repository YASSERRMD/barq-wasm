use barq_wasm::executor::BarqRuntime;
use clap::Parser;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path of the wasm file to run
    #[arg(short, long)]
    file: Option<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    match BarqRuntime::new() {
        Ok(runtime) => {
            if let Some(file) = args.file {
                eprintln!("error: cannot execute {file}");
            }
            match runtime.run() {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
