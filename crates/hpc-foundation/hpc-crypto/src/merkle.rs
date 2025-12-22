//! Merkle tree implementation for audit trails
//!
//! This module provides an append-only Merkle tree for creating
//! tamper-evident audit logs. It supports:
//!
//! - Construction from a list of leaves
//! - Incremental updates (append-only)
//! - Inclusion proof generation
//! - Proof verification
//!
//! # Security Properties
//!
//! - Append-only: New leaves can be added but existing ones cannot be modified
//! - Tamper-evident: Any modification to a leaf changes the root hash
//! - Efficient proofs: O(log n) proof size and verification time
//!
//! # Example
//!
//! ```
//! use hpc_crypto::merkle::MerkleTree;
//!
//! let leaves = vec![
//!     b"entry1".to_vec(),
//!     b"entry2".to_vec(),
//!     b"entry3".to_vec(),
//! ];
//!
//! let tree = MerkleTree::new(&leaves);
//! let root = tree.root();
//!
//! // Generate inclusion proof for first entry
//! let proof = tree.proof(0).unwrap();
//! assert!(proof.verify(&leaves[0], &root));
//! ```

use crate::hashing::{hash, HashValue};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during Merkle tree operations
#[derive(Error, Debug)]
pub enum MerkleError {
    #[error("Invalid proof")]
    InvalidProof,

    #[error("Invalid index: {0}")]
    InvalidIndex(usize),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Direction in a Merkle tree path
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Left,
    Right,
}

/// A node in the Merkle proof path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    hash: HashValue,
    direction: Direction,
}

/// Merkle inclusion proof
///
/// This proves that a specific leaf is included in the tree
/// with a given root hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    path: Vec<ProofNode>,
    leaf_index: usize,
}

impl MerkleProof {
    /// Get the proof path
    #[must_use]
    pub fn path(&self) -> &[ProofNode] {
        &self.path
    }

    /// Get the leaf index
    #[must_use]
    pub fn leaf_index(&self) -> usize {
        self.leaf_index
    }

    /// Verify that this proof validates the given leaf against the root
    ///
    /// # Returns
    ///
    /// `true` if the proof is valid, `false` otherwise
    #[must_use]
    pub fn verify(&self, leaf: &[u8], root: &HashValue) -> bool {
        let leaf_hash = hash(leaf);
        let mut current = leaf_hash;

        for node in &self.path {
            current = match node.direction {
                Direction::Left => combine_hashes(&node.hash, &current),
                Direction::Right => combine_hashes(&current, &node.hash),
            };
        }

        current == *root
    }

    /// Serialize the proof to JSON
    pub fn to_json(&self) -> Result<String, MerkleError> {
        serde_json::to_string(self).map_err(|e| MerkleError::SerializationError(e.to_string()))
    }

    /// Deserialize a proof from JSON
    pub fn from_json(s: &str) -> Result<Self, MerkleError> {
        serde_json::from_str(s).map_err(|e| MerkleError::SerializationError(e.to_string()))
    }
}

/// Merkle tree for audit trails
///
/// This is an append-only binary Merkle tree that supports
/// efficient inclusion proofs.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    leaves: Vec<HashValue>,
    layers: Vec<Vec<HashValue>>,
}

impl MerkleTree {
    /// Create a new Merkle tree from a list of leaves
    ///
    /// # Example
    ///
    /// ```
    /// use hpc_crypto::merkle::MerkleTree;
    ///
    /// let leaves = vec![b"leaf1".to_vec(), b"leaf2".to_vec()];
    /// let tree = MerkleTree::new(&leaves);
    /// ```
    #[must_use]
    pub fn new(leaves: &[Vec<u8>]) -> Self {
        let leaf_hashes: Vec<HashValue> = leaves.iter().map(|l| hash(l)).collect();

        if leaf_hashes.is_empty() {
            // Empty tree has a zero root
            return MerkleTree {
                leaves: vec![],
                layers: vec![vec![hash(b"")]],
            };
        }

        let layers = build_layers(&leaf_hashes);

        MerkleTree {
            leaves: leaf_hashes,
            layers,
        }
    }

    /// Get the root hash of the tree
    ///
    /// The root hash uniquely identifies the entire tree contents.
    #[must_use]
    pub fn root(&self) -> HashValue {
        self.layers.last().expect("tree should have layers")[0].clone()
    }

    /// Get the number of leaves in the tree
    #[must_use]
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Check if the tree is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Append a new leaf to the tree
    ///
    /// This rebuilds the tree with the new leaf included.
    pub fn append(&mut self, leaf: &[u8]) {
        let leaf_hash = hash(leaf);
        self.leaves.push(leaf_hash);
        self.layers = build_layers(&self.leaves);
    }

    /// Generate an inclusion proof for a specific leaf
    ///
    /// # Returns
    ///
    /// A proof that can be used to verify the leaf is in the tree,
    /// or `None` if the index is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use hpc_crypto::merkle::MerkleTree;
    ///
    /// let leaves = vec![b"leaf1".to_vec(), b"leaf2".to_vec()];
    /// let tree = MerkleTree::new(&leaves);
    /// let proof = tree.proof(0).unwrap();
    /// assert!(proof.verify(&leaves[0], &tree.root()));
    /// ```
    #[must_use]
    pub fn proof(&self, index: usize) -> Option<MerkleProof> {
        if index >= self.leaves.len() {
            return None;
        }

        let mut path = Vec::new();
        let mut current_index = index;

        // Walk up the tree from leaf to root
        for layer in &self.layers[..self.layers.len() - 1] {
            let is_right = current_index % 2 == 1;
            let sibling_index = if is_right {
                current_index - 1
            } else {
                current_index + 1
            };

            // If sibling exists, add it to the path
            if sibling_index < layer.len() {
                let direction = if is_right {
                    Direction::Left
                } else {
                    Direction::Right
                };

                path.push(ProofNode {
                    hash: layer[sibling_index].clone(),
                    direction,
                });
            }

            current_index /= 2;
        }

        Some(MerkleProof {
            path,
            leaf_index: index,
        })
    }
}

/// Combine two hashes to create a parent node hash
fn combine_hashes(left: &HashValue, right: &HashValue) -> HashValue {
    let mut combined = Vec::with_capacity(64);
    combined.extend_from_slice(left.as_bytes());
    combined.extend_from_slice(right.as_bytes());
    hash(&combined)
}

/// Build all layers of the Merkle tree from the leaves up to the root
fn build_layers(leaves: &[HashValue]) -> Vec<Vec<HashValue>> {
    if leaves.is_empty() {
        return vec![vec![hash(b"")]];
    }

    let mut layers = vec![leaves.to_vec()];

    while layers.last().expect("should have layers").len() > 1 {
        let current_layer = layers.last().expect("should have layers");
        let mut next_layer = Vec::new();

        for i in (0..current_layer.len()).step_by(2) {
            if i + 1 < current_layer.len() {
                // Pair exists, combine them
                next_layer.push(combine_hashes(&current_layer[i], &current_layer[i + 1]));
            } else {
                // Odd node at the end, promote it directly
                next_layer.push(current_layer[i].clone());
            }
        }

        layers.push(next_layer);
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree = MerkleTree::new(&[]);
        assert!(tree.is_empty());
        assert!(!tree.root().as_bytes().is_empty());
    }

    #[test]
    fn test_single_leaf() {
        let leaves = vec![b"leaf".to_vec()];
        let tree = MerkleTree::new(&leaves);
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_multiple_leaves() {
        let leaves = vec![
            b"leaf1".to_vec(),
            b"leaf2".to_vec(),
            b"leaf3".to_vec(),
        ];
        let tree = MerkleTree::new(&leaves);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_proof_generation() {
        let leaves = vec![b"leaf1".to_vec(), b"leaf2".to_vec()];
        let tree = MerkleTree::new(&leaves);
        let proof = tree.proof(0);
        assert!(proof.is_some());
    }

    #[test]
    fn test_proof_verification() {
        let leaves = vec![
            b"leaf1".to_vec(),
            b"leaf2".to_vec(),
            b"leaf3".to_vec(),
        ];
        let tree = MerkleTree::new(&leaves);
        let root = tree.root();

        for (i, leaf) in leaves.iter().enumerate() {
            let proof = tree.proof(i).unwrap();
            assert!(proof.verify(leaf, &root));
        }
    }

    #[test]
    fn test_proof_invalid_leaf() {
        let leaves = vec![b"leaf1".to_vec(), b"leaf2".to_vec()];
        let tree = MerkleTree::new(&leaves);
        let proof = tree.proof(0).unwrap();
        let root = tree.root();

        assert!(!proof.verify(b"wrong", &root));
    }

    #[test]
    fn test_append() {
        let initial = vec![b"leaf1".to_vec()];
        let mut tree = MerkleTree::new(&initial);
        let root1 = tree.root();

        tree.append(b"leaf2");
        let root2 = tree.root();

        assert_ne!(root1.as_bytes(), root2.as_bytes());
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_proof_serialization() {
        let leaves = vec![b"leaf1".to_vec(), b"leaf2".to_vec()];
        let tree = MerkleTree::new(&leaves);
        let proof = tree.proof(0).unwrap();

        let json = proof.to_json().unwrap();
        let restored = MerkleProof::from_json(&json).unwrap();

        let root = tree.root();
        assert!(restored.verify(&leaves[0], &root));
    }
}
