//! StratoSwarm CLI library

pub mod cli;
pub mod commands;
pub mod config;
pub mod error;
pub mod output;
pub mod shell;

pub use cli::Cli;
pub use commands::Commands;
pub use error::{CliError, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_export() {
        // Just check that the types are exported
        let _ = std::any::type_name::<Cli>();
        let _ = std::any::type_name::<Commands>();
        let _ = std::any::type_name::<CliError>();
    }
}
