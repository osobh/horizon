//! Checksum calculation algorithms

use crate::error::DisasterRecoveryResult;
use sha2::{Digest, Sha256, Sha512};
use std::io::Read;

use super::types::ChecksumAlgorithm;

/// Calculate checksum for data using specified algorithm
pub fn calculate_checksum(
    data: &[u8],
    algorithm: ChecksumAlgorithm,
) -> DisasterRecoveryResult<Vec<u8>> {
    match algorithm {
        ChecksumAlgorithm::SHA256 => {
            let mut hasher = Sha256::new();
            hasher.update(data);
            Ok(hasher.finalize().to_vec())
        }
        ChecksumAlgorithm::SHA512 => {
            let mut hasher = Sha512::new();
            hasher.update(data);
            Ok(hasher.finalize().to_vec())
        }
        ChecksumAlgorithm::CRC32 => {
            // CRC32 would require crc32fast crate
            // For now, use a simple hash based on data
            use std::hash::{Hash, Hasher};
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            data.hash(&mut hasher);
            let hash = hasher.finish() as u32;
            Ok(hash.to_le_bytes().to_vec())
        }
        ChecksumAlgorithm::MD5 => {
            // MD5 would require md5 crate
            // For now, use SHA256 truncated
            let mut hasher = Sha256::new();
            hasher.update(data);
            let full = hasher.finalize();
            Ok(full[..16].to_vec())
        }
        ChecksumAlgorithm::Blake2b => {
            // Blake2b would require blake2 crate
            // For now, use SHA512 as fallback
            let mut hasher = Sha512::new();
            hasher.update(data);
            Ok(hasher.finalize().to_vec())
        }
        ChecksumAlgorithm::XXHash => {
            // XXHash would require xxhash_rust crate
            // For now, use fast hash
            use std::hash::{Hash, Hasher};
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            data.hash(&mut hasher);
            let hash = hasher.finish();
            Ok(hash.to_le_bytes().to_vec())
        }
    }
}

/// Calculate checksum for a reader stream
pub fn calculate_checksum_streaming<R: Read>(
    mut reader: R,
    algorithm: ChecksumAlgorithm,
    buffer_size: usize,
) -> DisasterRecoveryResult<Vec<u8>> {
    let mut buffer = vec![0u8; buffer_size];
    
    match algorithm {
        ChecksumAlgorithm::SHA256 => {
            let mut hasher = Sha256::new();
            loop {
                let bytes_read = reader.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }
            Ok(hasher.finalize().to_vec())
        }
        ChecksumAlgorithm::SHA512 => {
            let mut hasher = Sha512::new();
            loop {
                let bytes_read = reader.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }
            Ok(hasher.finalize().to_vec())
        }
        _ => {
            // For other algorithms, read all data into memory
            // This is not ideal for large files
            let mut data = Vec::new();
            reader.read_to_end(&mut data)?;
            calculate_checksum(&data, algorithm)
        }
    }
}

/// Verify checksum matches expected value
pub fn verify_checksum(
    calculated: &[u8],
    expected: &[u8],
) -> bool {
    calculated == expected
}

/// Get recommended algorithm based on requirements
pub fn recommend_algorithm(
    performance_critical: bool,
    security_critical: bool,
) -> ChecksumAlgorithm {
    match (performance_critical, security_critical) {
        (true, false) => ChecksumAlgorithm::XXHash,
        (false, true) => ChecksumAlgorithm::SHA512,
        (true, true) => ChecksumAlgorithm::Blake2b,
        (false, false) => ChecksumAlgorithm::SHA256,
    }
}