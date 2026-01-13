#[cfg(test)]
mod test_cudarc_device_ptr {
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DeviceSlice};
    use std::sync::Arc;
    
    #[test]
    fn test_correct_device_ptr_usage() {
        let device = Arc::new(CudaContext::new(0).unwrap());
        // SAFETY: CudaDevice::alloc returns uninitialized GPU memory. This is safe
        // because we only test pointer access methods, not the memory contents.
        let mut slice: CudaSlice<u8> = unsafe { device.alloc(1024).unwrap() };
        
        // The correct way to get device pointer from CudaSlice
        // Based on cudarc API, we need to use as_device_ptr() method
        let ptr: *mut u8 = slice.as_mut_ptr();
        assert!(!ptr.is_null());
        
        // For immutable access
        let const_ptr: *const u8 = slice.as_ptr();
        assert!(!const_ptr.is_null());
    }
}