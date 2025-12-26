use std::fs;
use std::path::Path;

fn main() {
    let src_dir = Path::new("src");
    let evolution_file = src_dir.join("evolution.rs");
    let knowledge_file = src_dir.join("knowledge.rs");
    
    // Check and handle evolution.rs
    if evolution_file.exists() {
        println!("Found evolution.rs - attempting to handle...");
        
        // Try multiple approaches
        if let Err(e) = fs::remove_file(&evolution_file) {
            println!("Could not remove evolution.rs: {}", e);
            
            // Try renaming to hidden file
            let hidden = src_dir.join(".evolution.rs.conflict");
            if let Err(e) = fs::rename(&evolution_file, &hidden) {
                println!("Could not rename evolution.rs: {}", e);
                
                // Last resort - overwrite with empty module
                if let Err(e) = fs::write(&evolution_file, "// Module moved to evolution/mod.rs\n") {
                    println!("Could not overwrite evolution.rs: {}", e);
                }
            }
        } else {
            println!("Successfully removed evolution.rs");
        }
    }
    
    // Check and handle knowledge.rs
    if knowledge_file.exists() {
        println!("Found knowledge.rs - attempting to handle...");
        
        // Try multiple approaches  
        if let Err(e) = fs::remove_file(&knowledge_file) {
            println!("Could not remove knowledge.rs: {}", e);
            
            // Try renaming to hidden file
            let hidden = src_dir.join(".knowledge.rs.conflict");
            if let Err(e) = fs::rename(&knowledge_file, &hidden) {
                println!("Could not rename knowledge.rs: {}", e);
                
                // Last resort - overwrite with empty module
                if let Err(e) = fs::write(&knowledge_file, "// Module moved to knowledge/mod.rs\n") {
                    println!("Could not overwrite knowledge.rs: {}", e);
                }
            }
        } else {
            println!("Successfully removed knowledge.rs");
        }
    }
    
    // Check final state
    println!("\nFinal state:");
    println!("evolution.rs exists: {}", evolution_file.exists());
    println!("knowledge.rs exists: {}", knowledge_file.exists());
    println!("evolution/ directory exists: {}", src_dir.join("evolution").exists());
    println!("knowledge/ directory exists: {}", src_dir.join("knowledge").exists());
}