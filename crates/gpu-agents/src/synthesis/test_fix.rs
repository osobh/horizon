//! Test fixes for synthesis module
//! 
//! This provides minimal implementation to get tests passing

use super::*;

impl GpuSynthesisModule {
    /// CPU fallback for testing without GPU
    pub fn synthesize_cpu(&self, task: &SynthesisTask, input_ast: &AstNode) -> Result<String> {
        // Simple pattern matching
        if Self::matches_pattern(&task.pattern, input_ast) {
            // Extract bindings
            let bindings = Self::extract_bindings(&task.pattern, input_ast)?;
            
            // Expand template
            let result = Self::expand_template_cpu(&task.template, &bindings);
            Ok(result)
        } else {
            Ok(String::new())
        }
    }
    
    fn matches_pattern(pattern: &Pattern, ast: &AstNode) -> bool {
        // Check node type
        if pattern.node_type != ast.node_type {
            return false;
        }
        
        // Check value if not a variable
        if let Some(ref p_val) = pattern.value {
            if !p_val.starts_with('$') {
                if pattern.value != ast.value {
                    return false;
                }
            }
        }
        
        // Check children
        if pattern.children.len() != ast.children.len() {
            return false;
        }
        
        for (p_child, a_child) in pattern.children.iter().zip(ast.children.iter()) {
            if !Self::matches_pattern(p_child, a_child) {
                return false;
            }
        }
        
        true
    }
    
    fn extract_bindings(pattern: &Pattern, ast: &AstNode) -> Result<HashMap<String, String>> {
        let mut bindings = HashMap::new();
        Self::extract_bindings_recursive(pattern, ast, &mut bindings)?;
        Ok(bindings)
    }
    
    fn extract_bindings_recursive(pattern: &Pattern, ast: &AstNode, bindings: &mut HashMap<String, String>) -> Result<()> {
        // Extract variable bindings
        if let Some(ref p_val) = pattern.value {
            if p_val.starts_with('$') {
                if let Some(ref a_val) = ast.value {
                    bindings.insert(p_val.clone(), a_val.clone());
                }
            }
        }
        
        // Process children
        for (p_child, a_child) in pattern.children.iter().zip(ast.children.iter()) {
            Self::extract_bindings_recursive(p_child, a_child, bindings)?;
        }
        
        Ok(())
    }
    
    fn expand_template_cpu(template: &Template, bindings: &HashMap<String, String>) -> String {
        let mut result = String::new();
        
        for token in &template.tokens {
            match token {
                Token::Literal(s) => result.push_str(s),
                Token::Variable(var) => {
                    if let Some(value) = bindings.get(var) {
                        result.push_str(value);
                    } else {
                        result.push_str(var);
                    }
                }
            }
        }
        
        result
    }
}

impl pattern::GpuPatternMatcher {
    /// CPU fallback for pattern matching
    pub fn match_pattern_cpu(&self, pattern: &Pattern, ast: &AstNode) -> Result<Vec<Match>> {
        let mut matches = Vec::new();
        Self::find_matches_recursive(pattern, ast, 0, &mut matches)?;
        Ok(matches)
    }
    
    fn find_matches_recursive(pattern: &Pattern, ast: &AstNode, node_id: usize, matches: &mut Vec<Match>) -> Result<()> {
        if GpuSynthesisModule::matches_pattern(pattern, ast) {
            let bindings = GpuSynthesisModule::extract_bindings(pattern, ast)?;
            matches.push(Match {
                node_id,
                bindings,
            });
        }
        
        // Check children
        for (i, child) in ast.children.iter().enumerate() {
            Self::find_matches_recursive(pattern, child, node_id * 10 + i + 1, matches)?;
        }
        
        Ok(())
    }
}

impl template::GpuTemplateExpander {
    /// Get CPU result for testing
    pub fn expand_template_test(&self, template: &Template, bindings: &HashMap<String, String>) -> Result<String> {
        // Use CPU implementation
        Ok(self.expand_template_cpu(template, bindings))
    }
}

impl ast::GpuAstTransformer {
    /// CPU fallback for AST transformation
    pub fn transform_ast_cpu(&self, ast: &AstNode, rule: &TransformRule) -> Result<AstNode> {
        Self::transform_recursive(ast, rule)
    }
    
    fn transform_recursive(ast: &AstNode, rule: &TransformRule) -> Result<AstNode> {
        // Check if this node matches the pattern
        if GpuSynthesisModule::matches_pattern(&rule.pattern, ast) {
            // Apply transformation
            let bindings = GpuSynthesisModule::extract_bindings(&rule.pattern, ast)?;
            let transformed = Self::apply_transformation(&rule.replacement, &bindings)?;
            Ok(transformed)
        } else {
            // Transform children
            let mut result = ast.clone();
            result.children = ast.children.iter()
                .map(|child| Self::transform_recursive(child, rule))
                .collect::<Result<Vec<_>>>()?;
            Ok(result)
        }
    }
    
    fn apply_transformation(replacement: &AstNode, bindings: &HashMap<String, String>) -> Result<AstNode> {
        let mut result = replacement.clone();
        
        // Replace variables in the replacement
        if let Some(ref val) = result.value {
            if val.starts_with('$') {
                if let Some(binding) = bindings.get(val) {
                    result.value = Some(binding.clone());
                }
            }
        }
        
        // Process children
        result.children = replacement.children.iter()
            .map(|child| Self::apply_transformation(child, bindings))
            .collect::<Result<Vec<_>>>()?;
            
        Ok(result)
    }
}