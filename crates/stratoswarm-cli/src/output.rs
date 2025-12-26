//! Output formatting utilities for the CLI

use colored::Colorize;
use console::{style, Emoji};

static INFO: Emoji = Emoji("ℹ️ ", "");
static SUCCESS: Emoji = Emoji("✅", "✓");
static WARNING: Emoji = Emoji("⚠️ ", "!");
static ERROR: Emoji = Emoji("❌", "x");
static ROCKET: Emoji = Emoji("🚀", ">>");

/// Print an info message
pub fn info(message: &str) {
    println!("{} {}", style(INFO).blue(), message);
}

/// Print a success message
pub fn success(message: &str) {
    println!("{} {}", style(SUCCESS).green(), message.green());
}

/// Print a warning message  
pub fn warn(message: &str) {
    eprintln!("{} {}", style(WARNING).yellow(), message.yellow());
}

/// Print an error message
pub fn error(message: &str) {
    eprintln!("{} {}", style(ERROR).red(), message.red());
}

/// Print a launch message
pub fn launch(message: &str) {
    println!("{} {}", style(ROCKET).cyan(), message.cyan().bold());
}

/// Print a header
pub fn header(title: &str) {
    println!("\n{}\n", title.bold().underline());
}

/// Print a key-value pair
pub fn kv(key: &str, value: &str) {
    println!("{}: {}", key.bold(), value);
}

/// Print a bullet point
pub fn bullet(message: &str) {
    println!("  • {}", message);
}

/// Print a step in a process
pub fn step(number: usize, total: usize, message: &str) {
    println!("[{}/{}] {}", number, total, message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_functions() {
        // These should not panic
        info("Test info message");
        success("Test success message");
        warn("Test warning message");
        error("Test error message");
        launch("Test launch message");
        header("Test Header");
        kv("Key", "Value");
        bullet("Test bullet point");
        step(1, 3, "Test step");
    }
}
