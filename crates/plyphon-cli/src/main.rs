//! The `plyphon` binary: a SuperCollider-style OSC synthesis server and offline renderer.
//!
//! See [`cli`] for the command surface. Each subcommand handler returns `Result<(), String>` (the
//! repo's convention for example/CLI binaries); `main` prints any error and maps it to a failing
//! exit code.

mod audio;
mod bufsource;
mod cli;
mod defs;
mod options;
mod play;
mod render;
mod server;
mod transport;
mod wav;

use std::io;
use std::process::ExitCode;

use clap::{CommandFactory, Parser};

use crate::cli::{Cli, Command};

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Server(args) => server::run(args),
        Command::Render(args) => render::run(args),
        Command::Play(args) => play::run(args),
        Command::Devices => audio::list_devices(),
        Command::Completions(args) => {
            completions(args.shell);
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Print the completion script for `shell` to stdout.
fn completions(shell: clap_complete::Shell) {
    let mut command = Cli::command();
    clap_complete::generate(shell, &mut command, "plyphon", &mut io::stdout());
}
