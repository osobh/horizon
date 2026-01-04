//! Cryptographic primitives for HPC Platform
//!
//! This crate provides cryptographic operations for the HPC platform:
//!
//! - **Ed25519 signatures**: Digital signatures for authentication and integrity
//! - **Blake3 hashing**: Fast, secure cryptographic hashing
//! - **Merkle trees**: Tamper-evident audit logs with inclusion proofs
//!
//! # Security Considerations
//!
//! All cryptographic operations in this crate are designed with security in mind:
//!
//! - Constant-time operations where applicable (signature verification)
//! - Automatic zeroization of sensitive key material
//! - No panics in crypto code - all errors are returned as `Result`
//! - Uses well-audited cryptographic libraries (`ed25519-dalek`, `blake3`)
//!
//! # Examples
//!
//! ## Digital Signatures
//!
//! ```
//! use hpc_crypto::signing::KeyPair;
//!
//! // Generate a keypair
//! let keypair = KeyPair::generate();
//!
//! // Sign a message
//! let message = b"Hello, HPC!";
//! let signature = keypair.sign(message);
//!
//! // Verify the signature
//! assert!(keypair.public_key().verify(message, &signature).is_ok());
//! ```
//!
//! ## Hashing
//!
//! ```
//! use hpc_crypto::hashing::{hash, StreamHasher};
//!
//! // Simple hashing
//! let data = b"Hello, HPC!";
//! let hash_value = hash(data);
//! println!("Hash: {}", hash_value.to_hex());
//!
//! // Streaming hashing for large data
//! let mut hasher = StreamHasher::new();
//! hasher.update(b"Part 1, ");
//! hasher.update(b"Part 2");
//! let hash_value = hasher.finalize();
//! ```
//!
//! ## Merkle Trees for Audit Logs
//!
//! ```
//! use hpc_crypto::merkle::MerkleTree;
//!
//! // Create audit log entries
//! let entries = vec![
//!     b"user@example.com created job job-123".to_vec(),
//!     b"user@example.com started job job-123".to_vec(),
//!     b"user@example.com completed job job-123".to_vec(),
//! ];
//!
//! // Build Merkle tree
//! let tree = MerkleTree::new(&entries);
//! let root = tree.root();
//!
//! // Generate inclusion proof
//! let proof = tree.proof(1).unwrap();
//! assert!(proof.verify(&entries[1], &root));
//! ```

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]

pub mod hashing;
pub mod merkle;
pub mod signing;

// Re-export commonly used types
pub use hashing::{hash, hash_with_key, HashValue, StreamHasher};
pub use merkle::{MerkleProof, MerkleTree};
pub use signing::{KeyPair, PublicKey, Signature};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_sign_hash_integration() {
        // Sign a hash value
        let keypair = KeyPair::generate();
        let data = b"test data";
        let hash_value = hash(data);

        let signature = keypair.sign(hash_value.as_bytes());
        assert!(keypair
            .public_key()
            .verify(hash_value.as_bytes(), &signature)
            .is_ok());
    }

    #[test]
    fn test_merkle_with_signatures() {
        // Create signed entries in a Merkle tree
        let keypair = KeyPair::generate();
        let entries = [b"entry1".to_vec(), b"entry2".to_vec(), b"entry3".to_vec()];

        // Sign each entry
        let signatures: Vec<_> = entries.iter().map(|e| keypair.sign(e)).collect();

        // Build Merkle tree from signed entries
        let signed_entries: Vec<_> = entries
            .iter()
            .zip(signatures.iter())
            .map(|(entry, sig)| {
                let mut combined = entry.clone();
                combined.extend_from_slice(sig.as_bytes());
                combined
            })
            .collect();

        let tree = MerkleTree::new(&signed_entries);
        let root = tree.root();

        // Verify proof
        let proof = tree.proof(0).unwrap();
        assert!(proof.verify(&signed_entries[0], &root));
    }
}
