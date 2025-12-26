//! Binary to split large files in the codebase
//! Uses TDD approach to ensure correctness

use std::env;
use std::path::PathBuf;
use stratoswarm_zero_config::file_splitter::{FileSplitter, SplitConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <file_path>", args[0]);
        std::process::exit(1);
    }

    let file_path = PathBuf::from(&args[1]);

    if !file_path.exists() {
        eprintln!("Error: File not found: {:?}", file_path);
        std::process::exit(1);
    }

    println!("Splitting large file: {:?}", file_path);

    let config = SplitConfig {
        max_lines_per_file: 750,
        preserve_tests: true,
        create_mod_file: true,
    };

    let splitter = FileSplitter::new(config);

    match splitter.split_file(&file_path) {
        Ok(result) => {
            println!("\n✓ File split successfully!");
            println!(
                "  Original file: {:?} ({} lines)",
                result.original_file, result.line_count
            );
            println!("  Modules created: {}", result.modules_created.len());

            for module in &result.modules_created {
                println!("    - {} ({} lines)", module.name, module.line_count);
            }

            println!("\n  Main module: {:?}", result.main_module);
        }
        Err(e) => {
            eprintln!("Error splitting file: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
