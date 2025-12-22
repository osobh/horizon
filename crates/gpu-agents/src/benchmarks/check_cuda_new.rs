//! Check what CudaDevice::new actually returns

#[cfg(test)]
mod tests {
    #[test]
    fn check_cuda_device_new_signature() {
        // Let's see if we can NOT wrap in Arc
        use cudarc::driver::CudaDevice;
        
        if let Ok(device) = CudaDevice::new(0) {
            // If CudaDevice::new returns Arc<CudaDevice>, this won't compile
            let device_ref: &CudaDevice = &device;
            println!("Got reference to CudaDevice: {:p}", device_ref);
        }
    }
}