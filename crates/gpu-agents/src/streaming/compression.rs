//! GPU-accelerated compression algorithms

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

use super::GpuStreamKernel;

/// Compression algorithms
#[derive(Debug, Clone, Copy)]
pub enum CompressionAlgorithm {
    /// LZ4 compression
    Lz4,
    /// Run-length encoding
    Rle,
    /// Delta encoding
    Delta,
    /// Dictionary compression
    Dictionary,
}

/// GPU compressor
pub struct GpuCompressor {
    device: Arc<CudaContext>,
    algorithm: CompressionAlgorithm,
    /// Compression dictionary (if applicable)
    dictionary: Option<CudaSlice<u8>>,
    /// Temporary buffers
    _temp_buffer: CudaSlice<u8>,
}

impl GpuCompressor {
    /// Create new GPU compressor
    pub fn new(
        device: Arc<CudaContext>,
        algorithm: CompressionAlgorithm,
        buffer_size: usize,
    ) -> Result<Self> {
        // SAFETY: alloc returns uninitialized memory. temp_buffer is used as scratch
        // space by compression kernels and will be written before read.
        let stream = device.default_stream();
        let temp_buffer = unsafe { stream.alloc::<u8>(buffer_size)? };

        Ok(Self {
            device,
            algorithm,
            dictionary: None,
            _temp_buffer: temp_buffer,
        })
    }

    /// Set compression dictionary
    pub fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<()> {
        let stream = self.device.default_stream();
        let gpu_dict = stream.clone_htod(dictionary)?;
        self.dictionary = Some(gpu_dict);
        Ok(())
    }

    /// Compress data
    pub fn compress(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> Result<usize> {
        match self.algorithm {
            CompressionAlgorithm::Lz4 => self.compress_lz4(input, output, stream),
            CompressionAlgorithm::Rle => self.compress_rle(input, output, stream),
            CompressionAlgorithm::Delta => self.compress_delta(input, output, stream),
            CompressionAlgorithm::Dictionary => self.compress_dictionary(input, output, stream),
        }
    }

    /// LZ4 compression
    fn compress_lz4(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> Result<usize> {
        // Launch LZ4 compression kernel
        // SAFETY: All pointers are valid device pointers from CudaSlice references.
        // Input/output lengths match their allocation sizes. Stream is valid.
        let (input_ptr, _input_guard) = input.device_ptr(stream);
        let (output_ptr, _output_guard) = output.device_ptr(stream);
        unsafe {
            launch_lz4_compress(
                input_ptr as *const u8,
                output_ptr as *mut u8,
                input.len() as u32,
                output.len() as u32,
                std::ptr::null_mut(), // Use null for default stream
            );
        }

        // Return compressed size (would be computed by kernel)
        Ok(input.len() / 2) // Placeholder
    }

    /// Run-length encoding
    fn compress_rle(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> Result<usize> {
        // SAFETY: All pointers are valid device pointers from CudaSlice references.
        // Input/output lengths match their allocation sizes. Stream is valid.
        let (input_ptr, _input_guard) = input.device_ptr(stream);
        let (output_ptr, _output_guard) = output.device_ptr(stream);
        unsafe {
            launch_rle_compress(
                input_ptr as *const u8,
                output_ptr as *mut u8,
                input.len() as u32,
                output.len() as u32,
                std::ptr::null_mut(), // Use null for default stream
            );
        }
        Ok(input.len() / 3) // Placeholder
    }

    /// Delta encoding
    fn compress_delta(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> Result<usize> {
        // SAFETY: All pointers are valid device pointers from CudaSlice references.
        // Input/output lengths match their allocation sizes. Stream is valid.
        let (input_ptr, _input_guard) = input.device_ptr(stream);
        let (output_ptr, _output_guard) = output.device_ptr(stream);
        unsafe {
            launch_delta_compress(
                input_ptr as *const u8,
                output_ptr as *mut u8,
                input.len() as u32,
                output.len() as u32,
                std::ptr::null_mut(), // Use null for default stream
            );
        }
        Ok(input.len() * 3 / 4) // Placeholder
    }

    /// Dictionary compression
    fn compress_dictionary(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> Result<usize> {
        if let Some(ref dict) = self.dictionary {
            // SAFETY: All pointers are valid device pointers from CudaSlice references.
            // Input/output/dict lengths match their allocation sizes. Stream is valid.
            // Dictionary was set via set_dictionary() which copies to GPU.
            let (input_ptr, _input_guard) = input.device_ptr(stream);
            let (output_ptr, _output_guard) = output.device_ptr(stream);
            let (dict_ptr, _dict_guard) = dict.device_ptr(stream);
            unsafe {
                launch_dictionary_compress(
                    input_ptr as *const u8,
                    output_ptr as *mut u8,
                    dict_ptr as *const u8,
                    input.len() as u32,
                    output.len() as u32,
                    dict.len() as u32,
                    std::ptr::null_mut(), // Use null for default stream
                );
            }
            Ok(input.len() / 4) // Placeholder
        } else {
            Err(anyhow::anyhow!("Dictionary not set"))
        }
    }
}

impl GpuStreamKernel for GpuCompressor {
    fn name(&self) -> &str {
        match self.algorithm {
            CompressionAlgorithm::Lz4 => "lz4_compress",
            CompressionAlgorithm::Rle => "rle_compress",
            CompressionAlgorithm::Delta => "delta_compress",
            CompressionAlgorithm::Dictionary => "dict_compress",
        }
    }

    fn process(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.compress(input, output, stream)?;
        Ok(())
    }

    fn output_size(&self, input_size: usize) -> usize {
        // Conservative estimate - actual compression will vary
        match self.algorithm {
            CompressionAlgorithm::Lz4 => input_size,
            CompressionAlgorithm::Rle => input_size,
            CompressionAlgorithm::Delta => input_size,
            CompressionAlgorithm::Dictionary => input_size / 2,
        }
    }
}

/// GPU decompressor
pub struct GpuDecompressor {
    _device: Arc<CudaContext>,
    algorithm: CompressionAlgorithm,
    _dictionary: Option<CudaSlice<u8>>,
}

impl GpuDecompressor {
    /// Create new GPU decompressor
    pub fn new(device: Arc<CudaContext>, algorithm: CompressionAlgorithm) -> Result<Self> {
        Ok(Self {
            _device: device,
            algorithm,
            _dictionary: None,
        })
    }

    /// Decompress data
    pub fn decompress(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        compressed_size: usize,
        stream: &CudaStream,
    ) -> Result<usize> {
        match self.algorithm {
            CompressionAlgorithm::Lz4 => {
                // SAFETY: All pointers are valid device pointers from CudaSlice references.
                // compressed_size is passed from caller who knows the actual compressed data size.
                // Output length matches allocation. Stream is valid.
                let (input_ptr, _input_guard) = input.device_ptr(stream);
                let (output_ptr, _output_guard) = output.device_ptr(stream);
                unsafe {
                    launch_lz4_decompress(
                        input_ptr as *const u8,
                        output_ptr as *mut u8,
                        compressed_size as u32,
                        output.len() as u32,
                        std::ptr::null_mut(), // Use null for default stream
                    );
                }
                Ok(output.len()) // Placeholder
            }
            _ => Err(anyhow::anyhow!(
                "Decompression not implemented for {:?}",
                self.algorithm
            )),
        }
    }
}

// External CUDA kernel declarations
unsafe extern "C" {
    fn launch_lz4_compress(
        input: *const u8,
        output: *mut u8,
        input_size: u32,
        output_size: u32,
        stream: *mut std::ffi::c_void,
    );

    fn launch_lz4_decompress(
        input: *const u8,
        output: *mut u8,
        compressed_size: u32,
        output_size: u32,
        stream: *mut std::ffi::c_void,
    );

    fn launch_rle_compress(
        input: *const u8,
        output: *mut u8,
        input_size: u32,
        output_size: u32,
        stream: *mut std::ffi::c_void,
    );

    fn launch_delta_compress(
        input: *const u8,
        output: *mut u8,
        input_size: u32,
        output_size: u32,
        stream: *mut std::ffi::c_void,
    );

    fn launch_dictionary_compress(
        input: *const u8,
        output: *mut u8,
        dictionary: *const u8,
        input_size: u32,
        output_size: u32,
        dict_size: u32,
        stream: *mut std::ffi::c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_algorithm_names() -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(ctx) = CudaContext::new(0) {
            let lz4 = GpuCompressor::new(ctx.clone(), CompressionAlgorithm::Lz4, 1024)?;
            assert_eq!(lz4.name(), "lz4_compress");

            let rle = GpuCompressor::new(ctx.clone(), CompressionAlgorithm::Rle, 1024)?;
            assert_eq!(rle.name(), "rle_compress");
        }
        Ok(())
    }
}
