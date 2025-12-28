use barq_wasm::executor::BarqRuntime;
use barq_wasm::utils::logger;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the wasm file to run
    #[arg(short, long)]
    file: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _args = Args::parse();
    logger::init_logger();
    println!("Initializing Barq-WASM...");

    let _runtime = BarqRuntime::new()?;
    println!("Barq-WASM runtime initialized successfully.");

    Ok(())
}
