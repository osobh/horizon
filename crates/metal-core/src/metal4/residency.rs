//! Metal 4 residency set abstraction.
//!
//! Provides explicit GPU memory residency control that mirrors MTLResidencySet.
//! On Metal 4 (macOS 26+), this will wrap the native residency set.
//! On earlier systems, it tracks resources and applies them to encoders.
//!
//! # Why Residency Sets?
//!
//! Metal 4 removes automatic resource tracking for performance. Instead,
//! applications explicitly declare which resources the GPU needs:
//!
//! ```ignore
//! let mut residency_set = Metal4ResidencySet::new(&device)?;
//! residency_set.add_buffer(&input_buffer, ResourceUsage::Read)?;
//! residency_set.add_buffer(&output_buffer, ResourceUsage::Write)?;
//! residency_set.commit();
//! residency_set.request_residency()?;
//!
//! // In render/compute pass:
//! encoder.use_residency_set(&residency_set);
//! ```
//!
//! Benefits:
//! - GPU memory stays resident across multiple passes
//! - Explicit control over memory pressure
//! - Better performance than per-encoder useResource calls

use crate::buffer::MetalBuffer;
use crate::error::{MetalError, Result};
use crate::metal4::{Metal4Buffer, Metal4Device, Metal4Tensor};
use crate::tensor::MetalTensor;

use std::sync::Arc;

/// Resource usage flags for residency tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceUsage {
    /// Resource will only be read by the GPU.
    Read,
    /// Resource will only be written by the GPU.
    Write,
    /// Resource will be both read and written.
    ReadWrite,
}

impl ResourceUsage {
    /// Convert to Metal's resource usage flags (for useResource calls).
    pub fn as_metal_usage(&self) -> u32 {
        // These map to MTLResourceUsage values:
        // MTLResourceUsageRead = 1
        // MTLResourceUsageWrite = 2
        match self {
            ResourceUsage::Read => 1,
            ResourceUsage::Write => 2,
            ResourceUsage::ReadWrite => 3, // Read | Write
        }
    }
}

/// Tracks a resource and its usage in a residency set.
#[derive(Clone)]
pub struct TrackedResource {
    /// GPU address of the resource
    pub gpu_address: u64,
    /// How the resource will be used
    pub usage: ResourceUsage,
    /// Resource type for debugging
    pub resource_type: ResourceType,
}

/// Type of tracked resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    /// A Metal buffer
    Buffer,
    /// A Metal tensor
    Tensor,
    /// A Metal texture (future support)
    Texture,
}

/// Residency set for explicit GPU memory management.
///
/// On Metal 4: Wraps MTLResidencySet
/// On Metal 3: Tracks resources and applies them to encoders via useResource
///
/// # State Machine
///
/// The residency set has three states:
/// 1. **Building**: Resources can be added
/// 2. **Committed**: Resources are frozen, can request residency
/// 3. **Resident**: GPU memory is guaranteed resident
///
/// Call `commit()` to transition from Building to Committed.
/// Call `request_residency()` to transition from Committed to Resident.
pub struct Metal4ResidencySet {
    /// Reference to the device
    device: Arc<Metal4Device>,
    /// Tracked resources
    resources: Vec<TrackedResource>,
    /// Whether the set has been committed
    committed: bool,
    /// Whether residency has been requested
    resident: bool,
    /// Optional label for debugging
    label: Option<String>,
}

impl Metal4ResidencySet {
    /// Create a new residency set.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal 4 device
    ///
    /// # Errors
    ///
    /// Returns an error if the set cannot be created.
    pub fn new(device: &Arc<Metal4Device>) -> Result<Self> {
        Ok(Self {
            device: Arc::clone(device),
            resources: Vec::new(),
            committed: false,
            resident: false,
            label: None,
        })
    }

    /// Create a residency set with an estimated capacity.
    ///
    /// This pre-allocates space for the expected number of resources.
    pub fn with_capacity(device: &Arc<Metal4Device>, capacity: usize) -> Result<Self> {
        Ok(Self {
            device: Arc::clone(device),
            resources: Vec::with_capacity(capacity),
            committed: false,
            resident: false,
            label: None,
        })
    }

    /// Set a debug label for this residency set.
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }

    /// Add a buffer to the residency set.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to track
    /// * `usage` - How the buffer will be used
    ///
    /// # Errors
    ///
    /// Returns an error if the set has already been committed.
    pub fn add_buffer(&mut self, buffer: &Metal4Buffer, usage: ResourceUsage) -> Result<()> {
        if self.committed {
            return Err(MetalError::creation_failed(
                "ResidencySet",
                "Cannot add resources to a committed residency set",
            ));
        }

        self.resources.push(TrackedResource {
            gpu_address: buffer.gpu_address(),
            usage,
            resource_type: ResourceType::Buffer,
        });
        Ok(())
    }

    /// Add a tensor to the residency set.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to track
    /// * `usage` - How the tensor will be used
    ///
    /// # Errors
    ///
    /// Returns an error if the set has already been committed.
    pub fn add_tensor(&mut self, tensor: &Metal4Tensor, usage: ResourceUsage) -> Result<()> {
        if self.committed {
            return Err(MetalError::creation_failed(
                "ResidencySet",
                "Cannot add resources to a committed residency set",
            ));
        }

        self.resources.push(TrackedResource {
            gpu_address: tensor.gpu_address(),
            usage,
            resource_type: ResourceType::Tensor,
        });
        Ok(())
    }

    /// Add a raw GPU address to the residency set.
    ///
    /// Use this for advanced scenarios where you have the address directly.
    ///
    /// # Errors
    ///
    /// Returns an error if the set has already been committed.
    pub fn add_address(
        &mut self,
        address: u64,
        usage: ResourceUsage,
        resource_type: ResourceType,
    ) -> Result<()> {
        if self.committed {
            return Err(MetalError::creation_failed(
                "ResidencySet",
                "Cannot add resources to a committed residency set",
            ));
        }

        self.resources.push(TrackedResource {
            gpu_address: address,
            usage,
            resource_type,
        });
        Ok(())
    }

    /// Commit the residency set.
    ///
    /// After committing, no more resources can be added.
    /// Call this before `request_residency()`.
    pub fn commit(&mut self) {
        self.committed = true;
        // When Metal 4 is available:
        // self.raw.commit();
    }

    /// Request that all resources become GPU-resident.
    ///
    /// On Metal 4, this ensures the GPU memory is resident.
    /// On Metal 3, this is a no-op (memory is always resident).
    ///
    /// # Errors
    ///
    /// Returns an error if the set has not been committed.
    pub fn request_residency(&mut self) -> Result<()> {
        if !self.committed {
            return Err(MetalError::creation_failed(
                "ResidencySet",
                "Must commit residency set before requesting residency",
            ));
        }

        // On Metal 3: No-op, memory is always resident
        // On Metal 4: Will call self.raw.requestResidency()
        self.resident = true;
        Ok(())
    }

    /// End residency for all resources.
    ///
    /// Call this when you no longer need the resources to be GPU-resident.
    /// On Metal 3, this is a no-op.
    pub fn end_residency(&mut self) {
        self.resident = false;
        // When Metal 4 is available:
        // self.raw.endResidency();
    }

    /// Check if the set has been committed.
    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Check if residency has been requested.
    pub fn is_resident(&self) -> bool {
        self.resident
    }

    /// Get the number of tracked resources.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Get an iterator over the tracked resources.
    pub fn resources(&self) -> impl Iterator<Item = &TrackedResource> {
        self.resources.iter()
    }

    /// Get the label if set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get a reference to the device.
    pub fn device(&self) -> &Arc<Metal4Device> {
        &self.device
    }

    /// Reset the residency set for reuse.
    ///
    /// Clears all resources and resets the committed/resident state.
    pub fn reset(&mut self) {
        self.resources.clear();
        self.committed = false;
        self.resident = false;
    }

    /// Get resources filtered by usage type.
    pub fn resources_with_usage(
        &self,
        usage: ResourceUsage,
    ) -> impl Iterator<Item = &TrackedResource> {
        self.resources.iter().filter(move |r| r.usage == usage)
    }

    /// Get resources filtered by resource type.
    pub fn resources_of_type(
        &self,
        resource_type: ResourceType,
    ) -> impl Iterator<Item = &TrackedResource> {
        self.resources
            .iter()
            .filter(move |r| r.resource_type == resource_type)
    }
}

// SAFETY: Metal4ResidencySet is Send because:
// 1. Arc<Metal4Device> is Send + Sync
// 2. Vec<TrackedResource> is Send (TrackedResource contains only Send types)
// 3. Primitive fields are Send
unsafe impl Send for Metal4ResidencySet {}

// Note: Metal4ResidencySet is NOT Sync by default because:
// - add_buffer/add_tensor and other mutation methods are not atomic
// - Concurrent modification would cause data races
// Users should use external synchronization for concurrent access.

/// Builder pattern for creating residency sets with multiple resources.
pub struct ResidencySetBuilder {
    device: Arc<Metal4Device>,
    resources: Vec<TrackedResource>,
    label: Option<String>,
}

impl ResidencySetBuilder {
    /// Create a new builder.
    pub fn new(device: &Arc<Metal4Device>) -> Self {
        Self {
            device: Arc::clone(device),
            resources: Vec::new(),
            label: None,
        }
    }

    /// Add a buffer with the given usage.
    pub fn with_buffer(mut self, buffer: &Metal4Buffer, usage: ResourceUsage) -> Self {
        self.resources.push(TrackedResource {
            gpu_address: buffer.gpu_address(),
            usage,
            resource_type: ResourceType::Buffer,
        });
        self
    }

    /// Add a tensor with the given usage.
    pub fn with_tensor(mut self, tensor: &Metal4Tensor, usage: ResourceUsage) -> Self {
        self.resources.push(TrackedResource {
            gpu_address: tensor.gpu_address(),
            usage,
            resource_type: ResourceType::Tensor,
        });
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Build the residency set (uncommitted).
    pub fn build(self) -> Result<Metal4ResidencySet> {
        Ok(Metal4ResidencySet {
            device: self.device,
            resources: self.resources,
            committed: false,
            resident: false,
            label: self.label,
        })
    }

    /// Build and commit the residency set.
    pub fn build_committed(self) -> Result<Metal4ResidencySet> {
        let mut set = self.build()?;
        set.commit();
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residency_set_creation() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let set = Metal4ResidencySet::new(&device).expect("Failed to create residency set");

            assert_eq!(set.resource_count(), 0);
            assert!(set.is_empty());
            assert!(!set.is_committed());
            assert!(!set.is_resident());
        }
    }

    #[test]
    fn test_residency_set_add_buffer() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let mut set = Metal4ResidencySet::new(&device).expect("Failed to create residency set");

            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            set.add_buffer(&buffer, ResourceUsage::Read)
                .expect("Failed to add buffer");

            assert_eq!(set.resource_count(), 1);
            assert!(!set.is_empty());
        }
    }

    #[test]
    fn test_residency_set_commit_and_residency() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let mut set = Metal4ResidencySet::new(&device).expect("Failed to create residency set");

            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            set.add_buffer(&buffer, ResourceUsage::ReadWrite)
                .expect("Failed to add buffer");

            // Can't request residency before commit
            assert!(set.request_residency().is_err());

            set.commit();
            assert!(set.is_committed());

            // Can't add after commit
            let buffer2 = Metal4Buffer::new(&device, 512).expect("Failed to create buffer");
            assert!(set.add_buffer(&buffer2, ResourceUsage::Read).is_err());

            // Can request residency after commit
            set.request_residency()
                .expect("Failed to request residency");
            assert!(set.is_resident());
        }
    }

    #[test]
    fn test_residency_set_builder() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);

            let buffer1 = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            let buffer2 = Metal4Buffer::new(&device, 2048).expect("Failed to create buffer");

            let set = ResidencySetBuilder::new(&device)
                .with_buffer(&buffer1, ResourceUsage::Read)
                .with_buffer(&buffer2, ResourceUsage::Write)
                .with_label("TestResidencySet")
                .build_committed()
                .expect("Failed to build residency set");

            assert_eq!(set.resource_count(), 2);
            assert!(set.is_committed());
            assert_eq!(set.label(), Some("TestResidencySet"));
        }
    }

    #[test]
    fn test_residency_set_reset() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let mut set = Metal4ResidencySet::new(&device).expect("Failed to create residency set");

            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            set.add_buffer(&buffer, ResourceUsage::Read)
                .expect("Failed to add buffer");
            set.commit();
            set.request_residency()
                .expect("Failed to request residency");

            assert!(set.is_committed());
            assert!(set.is_resident());
            assert!(!set.is_empty());

            set.reset();

            assert!(!set.is_committed());
            assert!(!set.is_resident());
            assert!(set.is_empty());
        }
    }

    #[test]
    fn test_resource_usage_flags() {
        assert_eq!(ResourceUsage::Read.as_metal_usage(), 1);
        assert_eq!(ResourceUsage::Write.as_metal_usage(), 2);
        assert_eq!(ResourceUsage::ReadWrite.as_metal_usage(), 3);
    }

    #[test]
    fn test_resources_with_usage_filter() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let mut set = Metal4ResidencySet::new(&device).expect("Failed to create residency set");

            let buf1 = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            let buf2 = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            let buf3 = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");

            set.add_buffer(&buf1, ResourceUsage::Read).unwrap();
            set.add_buffer(&buf2, ResourceUsage::Write).unwrap();
            set.add_buffer(&buf3, ResourceUsage::Read).unwrap();

            let read_count = set.resources_with_usage(ResourceUsage::Read).count();
            let write_count = set.resources_with_usage(ResourceUsage::Write).count();

            assert_eq!(read_count, 2);
            assert_eq!(write_count, 1);
        }
    }
}
