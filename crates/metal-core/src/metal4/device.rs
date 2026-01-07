//! Metal 4 device abstraction.
//!
//! This module provides the Metal 4 device wrapper.
//! Metal 4 devices have enhanced capabilities for ML workloads.

use crate::backend::{DeviceInfo, MetalDevice};
use crate::error::Result;
use crate::metal3::Metal3Device;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// Metal 4 device wrapper.
///
/// Wraps an underlying Metal 3 device but exposes Metal 4 capabilities
/// when available. This allows incremental adoption of Metal 4 features.
pub struct Metal4Device {
    /// The underlying Metal 3 device (Metal 4 extends Metal 3)
    inner: Metal3Device,
    /// Cached device info
    info: DeviceInfo,
}

impl Metal4Device {
    /// Create a Metal 4 device from the system default GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available.
    pub fn system_default() -> Result<Self> {
        let inner = Metal3Device::system_default()?;
        let info = inner.info().clone();
        Ok(Self { inner, info })
    }

    /// Create a Metal 4 device from an existing MTLDevice.
    pub fn from_raw(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let inner = Metal3Device::from_raw(device);
        let info = inner.info().clone();
        Self { inner, info }
    }

    /// Get the underlying Metal 3 device.
    ///
    /// This is useful for operations that don't have Metal 4 equivalents yet.
    pub fn as_metal3_device(&self) -> &Metal3Device {
        &self.inner
    }

    /// Get the raw MTLDevice.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        self.inner.raw()
    }

    /// Check if this device supports Metal 4 features.
    ///
    /// Returns true only on macOS 26+ with Metal 4 capable hardware.
    pub fn supports_metal4(&self) -> bool {
        // Metal 4 is not yet available
        // When available, check for MTL4Compiler or similar marker
        false
    }

    /// Check if this device supports native MTLTensor.
    ///
    /// MTLTensor is a Metal 4 feature for first-class tensor support.
    pub fn supports_native_tensor(&self) -> bool {
        // Requires Metal 4
        false
    }

    /// Check if this device supports MTL4MachineLearningCommandEncoder.
    ///
    /// This encoder provides efficient neural network inference.
    pub fn supports_ml_encoder(&self) -> bool {
        // Requires Metal 4
        false
    }

    /// Check if this device supports MTL4ArgumentTable.
    ///
    /// Argument tables provide faster buffer binding than setBuffer calls.
    pub fn supports_argument_table(&self) -> bool {
        // Requires Metal 4
        false
    }

    /// Check if this device supports MTLResidencySet.
    ///
    /// Residency sets allow explicit control over GPU memory residency.
    pub fn supports_residency_set(&self) -> bool {
        // Requires Metal 4
        false
    }

    /// Get the maximum argument table capacity.
    ///
    /// Returns 0 if argument tables are not supported.
    pub fn max_argument_table_capacity(&self) -> usize {
        if self.supports_argument_table() {
            // FALLBACK: Returns conservative default until Metal 4 runtime query available.
            //
            // TODO(metal4): Query actual device limit via MTLDevice.argumentTableMaxSize
            // when running on macOS 15+ with Metal 4 support.
            //
            // The value 1024 is a safe minimum for Apple Silicon GPUs (M1+).
            1024
        } else {
            0
        }
    }
}

impl MetalDevice for Metal4Device {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal4_device_creation() {
        // This should succeed on any macOS system with Metal
        let result = Metal4Device::system_default();
        // May fail in CI without GPU
        if let Ok(device) = result {
            assert!(!device.name().is_empty());
            // Metal 4 features should not be available yet
            assert!(!device.supports_metal4());
            assert!(!device.supports_native_tensor());
            assert!(!device.supports_ml_encoder());
        }
    }

    #[test]
    fn test_metal4_feature_detection() {
        if let Ok(device) = Metal4Device::system_default() {
            // All Metal 4 features should be false until macOS 26
            assert!(!device.supports_metal4());
            assert!(!device.supports_native_tensor());
            assert!(!device.supports_ml_encoder());
            assert!(!device.supports_argument_table());
            assert!(!device.supports_residency_set());
            assert_eq!(device.max_argument_table_capacity(), 0);
        }
    }
}
