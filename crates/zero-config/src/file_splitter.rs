//! File Splitter Module for Large Files
//! TDD RED Phase - Create failing tests for file splitting

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Result of file splitting operation
#[derive(Debug, Serialize, Deserialize)]
pub struct SplitResult {
    pub original_file: PathBuf,
    pub line_count: usize,
    pub modules_created: Vec<ModuleInfo>,
    pub main_module: PathBuf,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub name: String,
    pub path: PathBuf,
    pub line_count: usize,
    pub exports: Vec<String>,
}

/// Configuration for splitting files
#[derive(Debug, Clone)]
pub struct SplitConfig {
    pub max_lines_per_file: usize,
    pub preserve_tests: bool,
    pub create_mod_file: bool,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            max_lines_per_file: 750,
            preserve_tests: true,
            create_mod_file: true,
        }
    }
}

/// File splitter for breaking large files into modules
pub struct FileSplitter {
    config: SplitConfig,
}

impl FileSplitter {
    pub fn new(config: SplitConfig) -> Self {
        Self { config }
    }

    /// Split a large file into smaller modules
    pub fn split_file(&self, file_path: &Path) -> Result<SplitResult, String> {
        // Check if file exists
        if !file_path.exists() {
            return Err(format!("File not found: {:?}", file_path));
        }

        // Read file content
        let content =
            fs::read_to_string(file_path).map_err(|e| format!("Failed to read file: {}", e))?;

        let line_count = content.lines().count();

        // Check if splitting is needed
        if line_count <= self.config.max_lines_per_file {
            return Ok(SplitResult {
                original_file: file_path.to_path_buf(),
                line_count,
                modules_created: vec![],
                main_module: file_path.to_path_buf(),
                success: true,
            });
        }

        // Analyze file structure
        let sections = self.analyze_file_structure(&content)?;

        // Plan module split
        let split_plan = self.plan_module_split(&sections)?;

        // Create modules
        let modules = self.create_modules(file_path, &split_plan)?;

        // Create main module file
        let main_module = self.create_main_module(file_path, &modules)?;

        Ok(SplitResult {
            original_file: file_path.to_path_buf(),
            line_count,
            modules_created: modules,
            main_module,
            success: true,
        })
    }

    /// Analyze file structure to identify logical sections
    fn analyze_file_structure(&self, content: &str) -> Result<Vec<FileSection>, String> {
        let mut sections = Vec::new();
        let mut current_section = FileSection::default();
        let mut in_impl_block = false;
        let mut in_test_module = false;

        for (line_num, line) in content.lines().enumerate() {
            // Detect test modules
            if line.contains("#[cfg(test)]") || line.contains("mod tests") {
                in_test_module = true;
            }

            // Detect impl blocks
            if line.starts_with("impl ") {
                in_impl_block = true;
                if !current_section.content.is_empty() {
                    sections.push(current_section);
                    current_section = FileSection::default();
                }
                current_section.section_type = if in_test_module {
                    SectionType::Tests
                } else {
                    SectionType::Implementation
                };
            }

            // Detect struct/enum definitions
            if line.starts_with("pub struct ")
                || line.starts_with("struct ")
                || line.starts_with("pub enum ")
                || line.starts_with("enum ")
            {
                if !current_section.content.is_empty() {
                    sections.push(current_section);
                    current_section = FileSection::default();
                }
                current_section.section_type = SectionType::Types;
            }

            current_section.content.push(line.to_string());
            current_section.line_count += 1;

            // Check for block end
            if in_impl_block && line == "}" {
                in_impl_block = false;
                sections.push(current_section);
                current_section = FileSection::default();
            }
        }

        // Add remaining section
        if !current_section.content.is_empty() {
            sections.push(current_section);
        }

        Ok(sections)
    }

    /// Plan how to split sections into modules
    fn plan_module_split(&self, sections: &[FileSection]) -> Result<Vec<ModulePlan>, String> {
        let mut plans = Vec::new();
        let mut current_plan = ModulePlan::default();

        for section in sections {
            if current_plan.line_count + section.line_count > self.config.max_lines_per_file {
                if !current_plan.sections.is_empty() {
                    plans.push(current_plan);
                }
                current_plan = ModulePlan::default();
            }

            current_plan.sections.push(section.clone());
            current_plan.line_count += section.line_count;

            // Determine module name based on content
            if current_plan.name.is_empty() {
                current_plan.name = match section.section_type {
                    SectionType::Types => "types".to_string(),
                    SectionType::Implementation => "impl".to_string(),
                    SectionType::Tests => "tests".to_string(),
                    SectionType::Utils => "utils".to_string(),
                };
            }
        }

        if !current_plan.sections.is_empty() {
            plans.push(current_plan);
        }

        Ok(plans)
    }

    /// Create module files from split plan
    fn create_modules(
        &self,
        base_path: &Path,
        plans: &[ModulePlan],
    ) -> Result<Vec<ModuleInfo>, String> {
        let mut modules = Vec::new();

        // Create directory for modules
        let parent_dir = base_path.parent().ok_or("No parent directory")?;
        let file_stem = base_path.file_stem().ok_or("No file stem")?;
        let module_dir = parent_dir.join(file_stem);

        fs::create_dir_all(&module_dir)
            .map_err(|e| format!("Failed to create module directory: {}", e))?;

        for (i, plan) in plans.iter().enumerate() {
            let module_name = if plans.len() > 1 && plan.name == "impl" {
                format!("{}_{}", plan.name, i)
            } else {
                plan.name.clone()
            };

            let module_path = module_dir.join(format!("{}.rs", module_name));
            let mut module_content = String::new();

            // Add module header
            module_content.push_str(&format!("//! {} module\n\n", module_name));

            // Add content
            for section in &plan.sections {
                for line in &section.content {
                    module_content.push_str(line);
                    module_content.push('\n');
                }
                module_content.push('\n');
            }

            // Write module file
            fs::write(&module_path, module_content)
                .map_err(|e| format!("Failed to write module: {}", e))?;

            modules.push(ModuleInfo {
                name: module_name,
                path: module_path,
                line_count: plan.line_count,
                exports: self.extract_exports(&plan.sections),
            });
        }

        Ok(modules)
    }

    /// Create main module file that re-exports submodules
    fn create_main_module(
        &self,
        original_path: &Path,
        modules: &[ModuleInfo],
    ) -> Result<PathBuf, String> {
        let mut content = String::new();

        // Add module declarations
        for module in modules {
            content.push_str(&format!("pub mod {};\n", module.name));
        }
        content.push('\n');

        // Add re-exports
        content.push_str("// Re-export main types\n");
        for module in modules {
            if !module.exports.is_empty() {
                content.push_str(&format!("pub use {}::{{", module.name));
                for (i, export) in module.exports.iter().enumerate() {
                    if i > 0 {
                        content.push_str(", ");
                    }
                    content.push_str(export);
                }
                content.push_str("};\n");
            }
        }

        // Create backup of original
        let backup_path = original_path.with_extension("rs.backup");
        fs::copy(original_path, &backup_path)
            .map_err(|e| format!("Failed to create backup: {}", e))?;

        // Write new main module
        fs::write(original_path, content)
            .map_err(|e| format!("Failed to write main module: {}", e))?;

        Ok(original_path.to_path_buf())
    }

    /// Extract public exports from sections
    fn extract_exports(&self, sections: &[FileSection]) -> Vec<String> {
        let mut exports = Vec::new();

        for section in sections {
            for line in &section.content {
                if line.starts_with("pub struct ") {
                    if let Some(name) = extract_name_from_line(line, "pub struct ") {
                        exports.push(name);
                    }
                } else if line.starts_with("pub enum ") {
                    if let Some(name) = extract_name_from_line(line, "pub enum ") {
                        exports.push(name);
                    }
                } else if line.starts_with("pub trait ") {
                    if let Some(name) = extract_name_from_line(line, "pub trait ") {
                        exports.push(name);
                    }
                } else if line.starts_with("pub fn ") {
                    if let Some(name) = extract_name_from_line(line, "pub fn ") {
                        exports.push(name);
                    }
                }
            }
        }

        exports
    }
}

#[derive(Debug, Clone, Default)]
struct FileSection {
    section_type: SectionType,
    content: Vec<String>,
    line_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum SectionType {
    Types,
    Implementation,
    Tests,
    Utils,
}

impl Default for SectionType {
    fn default() -> Self {
        SectionType::Utils
    }
}

#[derive(Debug, Default)]
struct ModulePlan {
    name: String,
    sections: Vec<FileSection>,
    line_count: usize,
}

fn extract_name_from_line(line: &str, prefix: &str) -> Option<String> {
    line.strip_prefix(prefix)
        .and_then(|s| {
            s.split(|c: char| c == ' ' || c == '<' || c == '(' || c == '{')
                .next()
        })
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_split_small_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("small.rs");

        let content = r#"
pub struct SmallStruct {
    field: String,
}

impl SmallStruct {
    pub fn new() -> Self {
        Self { field: String::new() }
    }
}
"#;
        fs::write(&file_path, content).unwrap();

        let splitter = FileSplitter::new(SplitConfig::default());
        let result = splitter.split_file(&file_path).unwrap();

        assert!(result.success);
        assert_eq!(result.modules_created.len(), 0); // No split needed
        assert!(result.line_count < 750);
    }

    #[test]
    fn test_split_large_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("large.rs");

        // Create a large file
        let mut content = String::new();

        // Add types section (400 lines)
        for i in 0..20 {
            content.push_str(&format!(
                r#"
#[derive(Debug, Clone)]
pub struct LargeStruct{} {{
    field1: String,
    field2: u64,
    field3: Vec<u8>,
    field4: Option<String>,
    field5: HashMap<String, String>,
}}

impl LargeStruct{} {{
    pub fn new() -> Self {{
        Self {{
            field1: String::new(),
            field2: 0,
            field3: Vec::new(),
            field4: None,
            field5: HashMap::new(),
        }}
    }}
}}
"#,
                i, i
            ));
        }

        // Add more implementations (400 lines)
        for i in 0..20 {
            content.push_str(&format!(
                r#"
impl Display for LargeStruct{} {{
    fn fmt(&self, f: &mut Formatter) -> Result {{
        write!(f, "LargeStruct{}")
    }}
}}
"#,
                i, i
            ));
        }

        // Add tests (400 lines)
        content.push_str(
            r#"
#[cfg(test)]
mod tests {
    use super::*;
    
"#,
        );
        for i in 0..20 {
            content.push_str(&format!(
                r#"
    #[test]
    fn test_struct{}() {{
        let s = LargeStruct{}::new();
        assert_eq!(s.field2, 0);
    }}
"#,
                i, i
            ));
        }
        content.push_str("}\n");

        fs::write(&file_path, content).unwrap();

        let splitter = FileSplitter::new(SplitConfig::default());
        let result = splitter.split_file(&file_path).unwrap();

        assert!(result.success);
        assert!(result.modules_created.len() > 0); // Should be split

        // Verify all modules are under 750 lines
        for module in &result.modules_created {
            assert!(
                module.line_count <= 750,
                "Module {} has {} lines, exceeding limit",
                module.name,
                module.line_count
            );
        }
    }

    #[test]
    fn test_preserve_test_modules() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("with_tests.rs");

        let content = r#"
pub struct MyStruct {
    value: i32,
}

impl MyStruct {
    pub fn new(value: i32) -> Self {
        Self { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new() {
        let s = MyStruct::new(42);
        assert_eq!(s.value, 42);
    }
}
"#;

        fs::write(&file_path, content).unwrap();

        let config = SplitConfig {
            max_lines_per_file: 10, // Force split
            preserve_tests: true,
            create_mod_file: true,
        };

        let splitter = FileSplitter::new(config);
        let result = splitter.split_file(&file_path).unwrap();

        assert!(result.success);

        // Check that tests are in a separate module
        let test_module = result.modules_created.iter().find(|m| m.name == "tests");
        assert!(test_module.is_some(), "Tests module should be preserved");
    }
}
