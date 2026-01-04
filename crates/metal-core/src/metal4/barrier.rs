//! Metal 4 barrier system.
//!
//! Metal 4 removes automatic resource tracking, requiring explicit barriers
//! for synchronization between GPU operations. This module provides types
//! for building and managing barriers.
//!
//! # Why Explicit Barriers?
//!
//! In Metal 4, all resources are "untracked" by default for performance.
//! Applications must explicitly synchronize access:
//!
//! ```ignore
//! // Producer pass writes to buffer
//! let barrier = BarrierBuilder::new()
//!     .scope(BarrierScope::Buffers)
//!     .produces(&output_buffer)
//!     .build();
//!
//! encoder.insert_barrier(&barrier);
//!
//! // Consumer pass reads from buffer
//! let barrier = BarrierBuilder::new()
//!     .scope(BarrierScope::Buffers)
//!     .consumes(&output_buffer)
//!     .build();
//!
//! encoder.insert_barrier(&barrier);
//! ```
//!
//! # Barrier Scopes
//!
//! - `Buffers`: Synchronize buffer access
//! - `Textures`: Synchronize texture access
//! - `RenderTargets`: Synchronize render target access
//! - `All`: Full memory barrier

use crate::buffer::MetalBuffer;
use crate::command::BarrierScope;
use crate::metal4::{Metal4Buffer, Metal4Tensor};
use crate::tensor::MetalTensor;

/// Reference to a resource for barrier tracking.
#[derive(Clone, Debug)]
pub struct ResourceRef {
    /// GPU address of the resource
    pub gpu_address: u64,
    /// Type of resource
    pub resource_type: ResourceType,
    /// Optional label for debugging
    pub label: Option<String>,
}

impl ResourceRef {
    /// Create a resource reference from a buffer.
    pub fn from_buffer(buffer: &Metal4Buffer) -> Self {
        Self {
            gpu_address: buffer.gpu_address(),
            resource_type: ResourceType::Buffer,
            label: None,
        }
    }

    /// Create a resource reference from a tensor.
    pub fn from_tensor(tensor: &Metal4Tensor) -> Self {
        Self {
            gpu_address: tensor.gpu_address(),
            resource_type: ResourceType::Tensor,
            label: None,
        }
    }

    /// Create a resource reference from a raw GPU address.
    pub fn from_address(address: u64, resource_type: ResourceType) -> Self {
        Self {
            gpu_address: address,
            resource_type,
            label: None,
        }
    }

    /// Add a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Type of resource being tracked in a barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// A Metal buffer
    Buffer,
    /// A Metal tensor
    Tensor,
    /// A Metal texture
    Texture,
    /// A render target (texture used as render attachment)
    RenderTarget,
}

/// Describes a memory barrier between GPU operations.
///
/// Barriers ensure proper ordering of read and write operations.
/// Use `BarrierBuilder` for convenient construction.
#[derive(Clone, Debug, Default)]
pub struct Barrier {
    /// Scope of the barrier
    pub scope: BarrierScope,
    /// Resources that this barrier waits for (consumed/read after barrier)
    pub before_resources: Vec<ResourceRef>,
    /// Resources that this barrier signals (produced/written before barrier)
    pub after_resources: Vec<ResourceRef>,
}

impl Barrier {
    /// Create a new empty barrier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a barrier with the given scope.
    pub fn with_scope(scope: BarrierScope) -> Self {
        Self {
            scope,
            ..Default::default()
        }
    }

    /// Create a simple buffer barrier (most common case).
    pub fn buffers() -> Self {
        Self::with_scope(BarrierScope::Buffers)
    }

    /// Create a simple texture barrier.
    pub fn textures() -> Self {
        Self::with_scope(BarrierScope::Textures)
    }

    /// Create a full memory barrier.
    pub fn all() -> Self {
        Self::with_scope(BarrierScope::All)
    }

    /// Check if this barrier has any tracked resources.
    pub fn has_resources(&self) -> bool {
        !self.before_resources.is_empty() || !self.after_resources.is_empty()
    }

    /// Get the total number of tracked resources.
    pub fn resource_count(&self) -> usize {
        self.before_resources.len() + self.after_resources.len()
    }
}

/// Builder for creating barriers with a fluent API.
///
/// # Example
///
/// ```ignore
/// let barrier = BarrierBuilder::new()
///     .scope(BarrierScope::Buffers)
///     .produces(&output_buffer)  // Written before this barrier
///     .consumes(&input_buffer)   // Read after this barrier
///     .build();
/// ```
#[derive(Default)]
pub struct BarrierBuilder {
    scope: BarrierScope,
    before_resources: Vec<ResourceRef>,
    after_resources: Vec<ResourceRef>,
}

impl BarrierBuilder {
    /// Create a new barrier builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the barrier scope.
    pub fn scope(mut self, scope: BarrierScope) -> Self {
        self.scope = scope;
        self
    }

    /// Add a buffer that is produced (written) before this barrier.
    ///
    /// The barrier will wait for all writes to this buffer to complete.
    pub fn produces_buffer(mut self, buffer: &Metal4Buffer) -> Self {
        self.after_resources.push(ResourceRef::from_buffer(buffer));
        self
    }

    /// Add a tensor that is produced (written) before this barrier.
    pub fn produces_tensor(mut self, tensor: &Metal4Tensor) -> Self {
        self.after_resources.push(ResourceRef::from_tensor(tensor));
        self
    }

    /// Add a resource reference that is produced before this barrier.
    pub fn produces(mut self, resource: ResourceRef) -> Self {
        self.after_resources.push(resource);
        self
    }

    /// Add a buffer that is consumed (read) after this barrier.
    ///
    /// The barrier ensures this buffer is safe to read.
    pub fn consumes_buffer(mut self, buffer: &Metal4Buffer) -> Self {
        self.before_resources.push(ResourceRef::from_buffer(buffer));
        self
    }

    /// Add a tensor that is consumed (read) after this barrier.
    pub fn consumes_tensor(mut self, tensor: &Metal4Tensor) -> Self {
        self.before_resources.push(ResourceRef::from_tensor(tensor));
        self
    }

    /// Add a resource reference that is consumed after this barrier.
    pub fn consumes(mut self, resource: ResourceRef) -> Self {
        self.before_resources.push(resource);
        self
    }

    /// Add a buffer that is both read and written.
    ///
    /// This is a shorthand for calling both produces and consumes.
    pub fn read_write_buffer(self, buffer: &Metal4Buffer) -> Self {
        self.produces_buffer(buffer).consumes_buffer(buffer)
    }

    /// Build the barrier.
    pub fn build(self) -> Barrier {
        Barrier {
            scope: self.scope,
            before_resources: self.before_resources,
            after_resources: self.after_resources,
        }
    }
}

/// Tracks barriers that need to be inserted.
///
/// Use this to batch multiple barriers for efficiency.
pub struct BarrierBatch {
    barriers: Vec<Barrier>,
}

impl BarrierBatch {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self {
            barriers: Vec::new(),
        }
    }

    /// Add a barrier to the batch.
    pub fn add(&mut self, barrier: Barrier) {
        self.barriers.push(barrier);
    }

    /// Add a barrier using the builder pattern.
    pub fn add_with<F>(&mut self, f: F)
    where
        F: FnOnce(BarrierBuilder) -> Barrier,
    {
        self.barriers.push(f(BarrierBuilder::new()));
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.barriers.is_empty()
    }

    /// Get the number of barriers.
    pub fn len(&self) -> usize {
        self.barriers.len()
    }

    /// Get an iterator over the barriers.
    pub fn iter(&self) -> impl Iterator<Item = &Barrier> {
        self.barriers.iter()
    }

    /// Consume the batch and return the barriers.
    pub fn into_vec(self) -> Vec<Barrier> {
        self.barriers
    }

    /// Clear all barriers from the batch.
    pub fn clear(&mut self) {
        self.barriers.clear();
    }

    /// Merge all barriers in the batch into a single barrier.
    ///
    /// This combines all resources and uses the most inclusive scope.
    pub fn merge(&self) -> Barrier {
        let mut merged = Barrier::new();

        for barrier in &self.barriers {
            // Use the most inclusive scope
            merged.scope = match (merged.scope, barrier.scope) {
                (BarrierScope::All, _) | (_, BarrierScope::All) => BarrierScope::All,
                (BarrierScope::Buffers, BarrierScope::Textures)
                | (BarrierScope::Textures, BarrierScope::Buffers) => BarrierScope::All,
                (s, _) => s,
            };

            merged
                .before_resources
                .extend(barrier.before_resources.clone());
            merged
                .after_resources
                .extend(barrier.after_resources.clone());
        }

        merged
    }
}

impl Default for BarrierBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IntoIterator for &'a BarrierBatch {
    type Item = &'a Barrier;
    type IntoIter = std::slice::Iter<'a, Barrier>;

    fn into_iter(self) -> Self::IntoIter {
        self.barriers.iter()
    }
}

impl IntoIterator for BarrierBatch {
    type Item = Barrier;
    type IntoIter = std::vec::IntoIter<Barrier>;

    fn into_iter(self) -> Self::IntoIter {
        self.barriers.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_creation() {
        let barrier = Barrier::new();
        assert!(!barrier.has_resources());
        assert_eq!(barrier.resource_count(), 0);
    }

    #[test]
    fn test_barrier_with_scope() {
        let barrier = Barrier::buffers();
        assert_eq!(barrier.scope, BarrierScope::Buffers);

        let barrier = Barrier::textures();
        assert_eq!(barrier.scope, BarrierScope::Textures);

        let barrier = Barrier::all();
        assert_eq!(barrier.scope, BarrierScope::All);
    }

    #[test]
    fn test_barrier_builder() {
        if let Ok(device) = crate::metal4::Metal4Device::system_default() {
            let device = std::sync::Arc::new(device);
            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");

            let barrier = BarrierBuilder::new()
                .scope(BarrierScope::Buffers)
                .produces_buffer(&buffer)
                .consumes_buffer(&buffer)
                .build();

            assert_eq!(barrier.scope, BarrierScope::Buffers);
            assert!(barrier.has_resources());
            assert_eq!(barrier.resource_count(), 2);
            assert_eq!(barrier.before_resources.len(), 1);
            assert_eq!(barrier.after_resources.len(), 1);
        }
    }

    #[test]
    fn test_resource_ref() {
        if let Ok(device) = crate::metal4::Metal4Device::system_default() {
            let device = std::sync::Arc::new(device);
            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");

            let resource_ref = ResourceRef::from_buffer(&buffer).with_label("test_buffer");

            assert_eq!(resource_ref.resource_type, ResourceType::Buffer);
            assert_eq!(resource_ref.label, Some("test_buffer".to_string()));
            assert!(resource_ref.gpu_address > 0);
        }
    }

    #[test]
    fn test_barrier_batch() {
        let mut batch = BarrierBatch::new();
        assert!(batch.is_empty());

        batch.add(Barrier::buffers());
        batch.add(Barrier::textures());

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());

        let merged = batch.merge();
        assert_eq!(merged.scope, BarrierScope::All);
    }

    #[test]
    fn test_barrier_batch_add_with() {
        let mut batch = BarrierBatch::new();

        batch.add_with(|b| b.scope(BarrierScope::Buffers).build());

        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_read_write_buffer() {
        if let Ok(device) = crate::metal4::Metal4Device::system_default() {
            let device = std::sync::Arc::new(device);
            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");

            let barrier = BarrierBuilder::new()
                .scope(BarrierScope::Buffers)
                .read_write_buffer(&buffer)
                .build();

            // read_write adds to both before and after
            assert_eq!(barrier.before_resources.len(), 1);
            assert_eq!(barrier.after_resources.len(), 1);
        }
    }
}
