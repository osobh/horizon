//! Metal 3 device implementation.

use crate::backend::{DeviceInfo, GpuFamily, MetalDevice};
use crate::device::parse_gpu_family;
use crate::error::{MetalError, Result};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLCreateSystemDefaultDevice};

/// Metal 3 device wrapper.
pub struct Metal3Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    info: DeviceInfo,
}

// SAFETY: MTLDevice is thread-safe according to Apple's documentation
unsafe impl Send for Metal3Device {}
unsafe impl Sync for Metal3Device {}

impl Metal3Device {
    /// Get the system default Metal device.
    pub fn system_default() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or(MetalError::NoDevice)?;

        let info = Self::query_device_info(&device);

        Ok(Self { device, info })
    }

    /// Create a Metal3Device from an existing MTLDevice.
    pub fn from_raw(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let info = Self::query_device_info(&device);
        Self { device, info }
    }

    /// Query device information.
    fn query_device_info(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> DeviceInfo {
        let name = device.name().to_string();
        let gpu_family = parse_gpu_family(&name);

        // Query device limits
        let max_buffer_length = device.maxBufferLength() as u64;

        // Apple Silicon has unified memory
        let unified_memory = device.hasUnifiedMemory();

        // Get max threads per threadgroup
        // This varies by GPU family, but 1024 is common for Apple Silicon
        let max_threads_per_threadgroup = 1024;

        // Threadgroup memory limit
        let max_threadgroup_memory_length = 32768; // 32KB is typical

        DeviceInfo {
            name,
            unified_memory,
            max_buffer_length,
            max_threads_per_threadgroup,
            max_threadgroup_memory_length,
            gpu_family,
        }
    }

    /// Get the raw MTLDevice.
    pub fn raw(&self) -> &Retained<ProtocolObject<dyn MTLDevice>> {
        &self.device
    }

    /// Check if the device supports a specific GPU family.
    pub fn supports_family(&self, family: GpuFamily) -> bool {
        match (&self.info.gpu_family, family) {
            (GpuFamily::Apple(have), GpuFamily::Apple(need)) => *have >= need,
            (GpuFamily::Mac(have), GpuFamily::Mac(need)) => *have >= need,
            _ => false,
        }
    }
}

impl MetalDevice for Metal3Device {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }
}

impl std::fmt::Debug for Metal3Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metal3Device")
            .field("name", &self.info.name)
            .field("unified_memory", &self.info.unified_memory)
            .field("max_buffer_length", &self.info.max_buffer_length)
            .finish()
    }
}
