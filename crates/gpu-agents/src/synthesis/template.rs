//! GPU Template Expansion Implementation
//!
//! Parallel template expansion with variable substitution on GPU

use crate::synthesis::{Template, Token};
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use std::collections::HashMap;
use std::sync::Arc;

/// GPU Template Expander for parallel code generation
pub struct GpuTemplateExpander {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    template_buffer: CudaSlice<u8>,
    binding_buffer: CudaSlice<u8>,
    output_buffer: CudaSlice<u8>,
    max_output_size: usize,
}

impl GpuTemplateExpander {
    /// Create a new GPU template expander
    pub fn new(device: Arc<CudaContext>, max_output_size: usize) -> Result<Self> {
        let stream = device.default_stream();
        // Allocate GPU buffers
        // SAFETY: alloc returns uninitialized memory. template_buffer will be written
        // via memcpy_htod in expand_template() before the kernel reads it.
        let template_buffer = unsafe { stream.alloc::<u8>(max_output_size) }
            .context("Failed to allocate template buffer")?;
        // SAFETY: alloc returns uninitialized memory. binding_buffer will be written
        // via memcpy_htod in expand_template() before the kernel reads it.
        let binding_buffer = unsafe { stream.alloc::<u8>(max_output_size) }
            .context("Failed to allocate binding buffer")?;
        // SAFETY: alloc returns uninitialized memory. output_buffer will be cleared
        // to zeros in expand_template() before the kernel writes results.
        let output_buffer = unsafe { stream.alloc::<u8>(max_output_size) }
            .context("Failed to allocate output buffer")?;

        Ok(Self {
            device,
            stream,
            template_buffer,
            binding_buffer,
            output_buffer,
            max_output_size,
        })
    }

    /// Expand a template with variable bindings
    pub fn expand_template(
        &self,
        template: &Template,
        bindings: &HashMap<String, String>,
    ) -> Result<String> {
        // Encode template and bindings for GPU
        let template_data = self.encode_template(template)?;
        let binding_data = self.encode_bindings(bindings)?;

        // Copy to GPU
        self.stream
            .memcpy_htod(&template_data, &mut self.template_buffer.clone())?;
        self.stream
            .memcpy_htod(&binding_data, &mut self.binding_buffer.clone())?;

        // Clear output buffer
        let zeros = vec![0u8; self.max_output_size];
        self.stream
            .memcpy_htod(&zeros, &mut self.output_buffer.clone())?;

        // Launch template expansion kernel
        // SAFETY: All pointers are valid device pointers from CudaSlice allocations:
        // - template_buffer: populated via memcpy_htod above
        // - binding_buffer: populated via memcpy_htod above
        // - output_buffer: cleared to zeros above, kernel writes expanded result
        // - max_output_size matches the allocation size
        unsafe {
            let (template_ptr, _guard1) = self.template_buffer.device_ptr(&self.stream);
            let (binding_ptr, _guard2) = self.binding_buffer.device_ptr(&self.stream);
            let (output_ptr, _guard3) = self.output_buffer.device_ptr(&self.stream);
            crate::synthesis::launch_expand_templates(
                template_ptr as *const u8,
                binding_ptr as *const u8,
                output_ptr as *mut u8,
                1, // Single template for now
                self.max_output_size as u32,
            );
        }

        // Synchronize and get results
        self.stream.synchronize()?;

        let output: Vec<u8> = self.stream.clone_dtoh(&self.output_buffer)?;

        // Convert output to string
        let result = self.decode_output(&output)?;

        Ok(result)
    }

    /// Expand multiple templates in parallel
    pub fn expand_templates_batch(
        &self,
        tasks: &[(Template, HashMap<String, String>)],
    ) -> Result<Vec<String>> {
        let mut results = Vec::new();

        // Process each template
        for (template, bindings) in tasks {
            let expanded = self.expand_template(template, bindings)?;
            results.push(expanded);
        }

        Ok(results)
    }

    /// Encode template for GPU processing
    fn encode_template(&self, template: &Template) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();

        // Format: [num_tokens(4), token_data...]
        encoded.extend_from_slice(&(template.tokens.len() as u32).to_le_bytes());

        for token in &template.tokens {
            match token {
                Token::Literal(s) => {
                    encoded.push(0); // Token type: literal
                    encoded.extend_from_slice(&(s.len() as u32).to_le_bytes());
                    encoded.extend_from_slice(s.as_bytes());
                }
                Token::Variable(s) => {
                    encoded.push(1); // Token type: variable
                    encoded.extend_from_slice(&(s.len() as u32).to_le_bytes());
                    encoded.extend_from_slice(s.as_bytes());
                }
            }
        }

        Ok(encoded)
    }

    /// Encode bindings for GPU processing
    fn encode_bindings(&self, bindings: &HashMap<String, String>) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();

        // Format: [num_bindings(4), binding_data...]
        encoded.extend_from_slice(&(bindings.len() as u32).to_le_bytes());

        for (var, value) in bindings {
            // Variable name
            encoded.extend_from_slice(&(var.len() as u32).to_le_bytes());
            encoded.extend_from_slice(var.as_bytes());

            // Value
            encoded.extend_from_slice(&(value.len() as u32).to_le_bytes());
            encoded.extend_from_slice(value.as_bytes());
        }

        Ok(encoded)
    }

    /// Decode output from GPU
    fn decode_output(&self, output: &[u8]) -> Result<String> {
        // Find null terminator
        let end = output.iter().position(|&b| b == 0).unwrap_or(output.len());

        // Convert to string
        let result = String::from_utf8(output[..end].to_vec())
            .context("Failed to decode output as UTF-8")?;

        Ok(result)
    }

    /// Simple CPU-based template expansion for fallback/testing
    pub fn expand_template_cpu(
        &self,
        template: &Template,
        bindings: &HashMap<String, String>,
    ) -> String {
        let mut result = String::new();

        for token in &template.tokens {
            match token {
                Token::Literal(s) => result.push_str(s),
                Token::Variable(var) => {
                    if let Some(value) = bindings.get(var) {
                        result.push_str(value);
                    } else {
                        result.push_str(var); // Keep variable if no binding
                    }
                }
            }
        }

        result
    }
}

impl Drop for GpuTemplateExpander {
    fn drop(&mut self) {
        // Buffers are automatically freed when CudaSlice is dropped
    }
}
