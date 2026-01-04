#[cfg(test)]
mod test_device_ptr_fixes {
    use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
    use std::sync::Arc;

    #[test]
    fn test_device_ptr_returns_pointer_not_u64() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        // SAFETY: CudaDevice::alloc returns uninitialized GPU memory. This is safe
        // because we only test pointer validity, not the memory contents.
        let mut slice: CudaSlice<u8> = unsafe { device.alloc(1024).unwrap() };

        // DevicePtr trait should give us a raw pointer, not u64
        let ptr: *mut u8 = slice.device_ptr_mut();
        assert!(!ptr.is_null());

        // For const access
        let const_ptr: *const u8 = slice.device_ptr();
        assert!(!const_ptr.is_null());
    }

    #[test]
    fn test_cuda_device_memset() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        // SAFETY: alloc_zeros returns zero-initialized GPU memory, which is
        // always safe to use immediately.
        let mut slice: CudaSlice<u8> = unsafe { device.alloc_zeros(1024).unwrap() };

        // CudaDevice should have memset functionality
        // Using launch_memset or similar
        device.memset(&mut slice, 42u8)?;
    }
}
