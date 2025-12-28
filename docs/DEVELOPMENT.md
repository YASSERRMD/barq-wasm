# Development Guide

## Setup
1. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Clone the repository: `git clone https://github.com/YASSERRMD/barq-wasm.git`

## Build Commands
- Standard build: `cargo build`
- Release build: `cargo build --release`
- Clean: `cargo clean`

## Test Commands
- Run all tests: `cargo test`
- Run integration tests: `cargo test --test integration_tests`
- Run clippy: `cargo clippy -- -D warnings`
- Run formatter check: `cargo fmt -- --check`

## Code Style
- Follow standard Rust conventions.
- All public functions should have documentation.
- No `unwrap()` in production code; use `anyhow` or `thiserror`.
- Keep modules focused and cohesive.
