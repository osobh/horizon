//! Advanced template operations for synthesis
//!
//! Provides enhanced template functionality including conditionals, loops, and functions

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Template variable for substitution
#[derive(Debug, Clone)]
pub struct TemplateVariable {
    pub name: String,
    pub value: String,
    pub var_type: VariableType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VariableType {
    String,
    Number,
    Boolean,
    List,
}

/// Template condition for conditional rendering
#[derive(Debug, Clone)]
pub struct TemplateCondition {
    pub variable: String,
    pub value: bool,
    pub operator: ConditionOperator,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
}

impl TemplateCondition {
    pub fn new(variable: &str, value: bool) -> Self {
        Self {
            variable: variable.to_string(),
            value,
            operator: ConditionOperator::Equals,
        }
    }
}

/// Template loop for iterative expansion
#[derive(Debug, Clone)]
pub struct TemplateLoop {
    pub variable: String,
    pub items: Vec<String>,
    pub index_var: Option<String>,
}

impl TemplateLoop {
    pub fn new(variable: &str, items: Vec<&str>) -> Self {
        Self {
            variable: variable.to_string(),
            items: items.iter().map(|s| s.to_string()).collect(),
            index_var: None,
        }
    }
}

/// Template function for transformations
pub struct TemplateFunction {
    pub name: String,
    pub func: Box<dyn Fn(&str) -> String + Send + Sync>,
}

impl TemplateFunction {
    pub fn new<F>(name: &str, func: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            func: Box::new(func),
        }
    }
}

/// Composite template with multiple parts
#[derive(Debug, Clone)]
pub struct CompositeTemplate {
    pub templates: HashMap<String, String>,
    pub order: Vec<String>,
}

impl CompositeTemplate {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn add_template(mut self, name: &str, template: &str) -> Self {
        self.templates
            .insert(name.to_string(), template.to_string());
        self.order.push(name.to_string());
        self
    }
}

/// Main template engine with advanced operations
pub struct TemplateEngine {
    device: Arc<CudaDevice>,
    functions: Arc<Mutex<HashMap<String, Box<dyn Fn(&str) -> String + Send + Sync>>>>,
    variable_buffer: Option<CudaSlice<u8>>,
    template_buffer: Option<CudaSlice<u8>>,
    output_buffer: Option<CudaSlice<u8>>,
    max_buffer_size: usize,
}

impl TemplateEngine {
    /// Create new template engine
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let max_buffer_size = 1024 * 1024; // 1MB default

        // Allocate GPU buffers
        let variable_buffer = unsafe { device.alloc::<u8>(max_buffer_size) }.ok();
        let template_buffer = unsafe { device.alloc::<u8>(max_buffer_size) }.ok();
        let output_buffer = unsafe { device.alloc::<u8>(max_buffer_size) }.ok();

        Ok(Self {
            device,
            functions: Arc::new(Mutex::new(HashMap::new())),
            variable_buffer,
            template_buffer,
            output_buffer,
            max_buffer_size,
        })
    }

    /// Substitute variables in template
    pub fn substitute_variables(
        &self,
        template: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String> {
        let mut result = template.to_string();

        // Simple variable substitution
        for (name, value) in variables {
            let pattern = format!("{{{{{}}}}}", name);
            result = result.replace(&pattern, value);
        }

        Ok(result)
    }

    /// Evaluate conditional template
    pub fn evaluate_condition(
        &self,
        condition: &TemplateCondition,
        if_template: &str,
        else_template: &str,
    ) -> Result<String> {
        // For now, simple boolean evaluation
        if condition.value {
            Ok(if_template.to_string())
        } else {
            Ok(else_template.to_string())
        }
    }

    /// Expand loop template
    pub fn expand_loop(&self, loop_def: &TemplateLoop, item_template: &str) -> Result<String> {
        let mut result = String::new();

        for (index, item) in loop_def.items.iter().enumerate() {
            let mut expanded = item_template.to_string();

            // Replace item variable
            expanded = expanded.replace("{{item}}", item);

            // Replace index if specified
            if let Some(ref index_var) = loop_def.index_var {
                expanded = expanded.replace(&format!("{{{{{}}}}}", index_var), &index.to_string());
            }

            result.push_str(&expanded);
            result.push('\n');
        }

        Ok(result.trim_end().to_string())
    }

    /// Register a template function
    pub fn register_function(&self, function: TemplateFunction) -> Result<()> {
        let mut functions = self.functions.lock()?;
        functions.insert(function.name, function.func);
        Ok(())
    }

    /// Apply functions in template
    pub fn apply_functions(
        &self,
        template: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String> {
        // First substitute variables
        let mut result = self.substitute_variables(template, variables)?;
        let functions = self.functions.lock()?;

        // Find function calls in template
        let re = regex::Regex::new(r"\{\{(\w+)\((\w+)\)\}\}")?;

        // Keep replacing until no more function calls
        loop {
            let mut changed = false;
            for cap in re.captures_iter(&result.clone()) {
                let func_name = &cap[1];
                let var_name = &cap[2];

                if let (Some(func), Some(value)) =
                    (functions.get(func_name), variables.get(var_name))
                {
                    let transformed = func(value);
                    result = result.replace(&cap[0], &transformed);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        Ok(result)
    }

    /// Render composite template
    pub fn render_composite(
        &self,
        composite: &CompositeTemplate,
        variables: &HashMap<String, String>,
    ) -> Result<String> {
        let mut result = String::new();

        for name in &composite.order {
            if let Some(template) = composite.templates.get(name) {
                let rendered = self.substitute_variables(template, variables)?;
                result.push_str(&rendered);
                result.push('\n');
            }
        }

        Ok(result.trim_end().to_string())
    }

    /// GPU-accelerated batch template expansion
    pub fn gpu_batch_expand(
        &self,
        templates: &[&str],
        batch_vars: &[HashMap<String, String>],
    ) -> Result<Vec<String>> {
        if templates.len() != batch_vars.len() {
            return Err(anyhow!("Template and variable counts must match"));
        }

        let mut results = Vec::new();

        // For now, process sequentially (GPU acceleration would be implemented here)
        for (template, vars) in templates.iter().zip(batch_vars.iter()) {
            let result = self.substitute_variables(template, vars)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Encode templates for GPU processing
    fn encode_templates(&self, templates: &[&str]) -> Vec<u8> {
        let mut encoded = Vec::new();

        // Number of templates
        encoded.extend_from_slice(&(templates.len() as u32).to_le_bytes());

        // Each template
        for template in templates {
            encoded.extend_from_slice(&(template.len() as u32).to_le_bytes());
            encoded.extend_from_slice(template.as_bytes());
        }

        encoded
    }

    /// Encode variables for GPU processing
    fn encode_variables(&self, variables: &[HashMap<String, String>]) -> Vec<u8> {
        let mut encoded = Vec::new();

        // Number of variable sets
        encoded.extend_from_slice(&(variables.len() as u32).to_le_bytes());

        // Each variable set
        for vars in variables {
            encoded.extend_from_slice(&(vars.len() as u32).to_le_bytes());

            for (name, value) in vars {
                // Name
                encoded.extend_from_slice(&(name.len() as u32).to_le_bytes());
                encoded.extend_from_slice(name.as_bytes());

                // Value
                encoded.extend_from_slice(&(value.len() as u32).to_le_bytes());
                encoded.extend_from_slice(value.as_bytes());
            }
        }

        encoded
    }
}

/// Built-in template functions
pub fn register_builtin_functions(engine: &TemplateEngine) -> Result<()> {
    // Uppercase function
    engine.register_function(TemplateFunction::new("uppercase", |s| s.to_uppercase()))?;

    // Lowercase function
    engine.register_function(TemplateFunction::new("lowercase", |s| s.to_lowercase()))?;

    // Capitalize function
    engine.register_function(TemplateFunction::new("capitalize", |s| {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    }))?;

    // Length function
    engine.register_function(TemplateFunction::new("length", |s| s.len().to_string()))?;

    // Reverse function
    engine.register_function(TemplateFunction::new("reverse", |s| {
        s.chars().rev().collect()
    }))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_substitution() -> Result<(), Box<dyn std::error::Error>>  {
        let device = CudaDevice::new(0)?;
        let engine = TemplateEngine::new(device)?;

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("age".to_string(), "30".to_string());

        let template = "Hello {{name}}, you are {{age}} years old.";
        let result = engine.substitute_variables(template, &vars)?;

        assert_eq!(result, "Hello Alice, you are 30 years old.");
    }

    #[test]
    fn test_loop_expansion() -> Result<(), Box<dyn std::error::Error>>  {
        let device = CudaDevice::new(0)?;
        let engine = TemplateEngine::new(device)?;

        let loop_def = TemplateLoop::new("fruits", vec!["apple", "banana", "orange"]);
        let template = "- {{item}}";

        let result = engine.expand_loop(&loop_def, template)?;
        assert_eq!(result, "- apple\n- banana\n- orange");
    }
}
