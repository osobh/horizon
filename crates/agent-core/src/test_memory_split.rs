//! TDD test for memory.rs file splitting

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_memory_split() {
        // RED phase: Test that memory.rs should be split into smaller modules
        let memory_path = Path::new("src/memory.rs");

        if memory_path.exists() {
            let content = fs::read_to_string(memory_path).unwrap();
            let line_count = content.lines().count();

            // Should be under 750 lines after split
            assert!(
                line_count <= 750,
                "memory.rs has {} lines, should be under 750",
                line_count
            );
        }

        // Verify sub-modules exist
        let expected_modules = vec![
            "memory/types.rs",
            "memory/entry.rs",
            "memory/store.rs",
            "memory/system.rs",
            "memory/search.rs",
        ];

        for module in expected_modules {
            let module_path = Path::new("src").join(module);
            assert!(module_path.exists(), "Expected module {} to exist", module);

            // Each module should be under 750 lines
            if module_path.exists() {
                let content = fs::read_to_string(&module_path).unwrap();
                let line_count = content.lines().count();
                assert!(
                    line_count <= 750,
                    "Module {} has {} lines, should be under 750",
                    module,
                    line_count
                );
            }
        }
    }
}
