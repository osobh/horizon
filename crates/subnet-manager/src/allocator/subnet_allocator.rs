//! CIDR block allocator for subnet management
//!
//! Allocates non-overlapping CIDR blocks from a parent address space.
//! Uses a BTreeMap for efficient O(log n) operations and gap detection.

use crate::{Error, Result};
use ipnet::Ipv4Net;
use std::collections::BTreeMap;
use std::net::Ipv4Addr;

/// Trait for CIDR block allocation
pub trait CidrAllocator: Send + Sync {
    /// Allocate a CIDR block of the given prefix length
    fn allocate(&mut self, prefix_len: u8) -> Result<Ipv4Net>;

    /// Reserve a specific CIDR block
    fn reserve(&mut self, cidr: Ipv4Net) -> Result<()>;

    /// Release a previously allocated CIDR block
    fn release(&mut self, cidr: Ipv4Net) -> Result<()>;

    /// Check if a CIDR block is allocated
    fn is_allocated(&self, cidr: &Ipv4Net) -> bool;

    /// Check if a CIDR block overlaps with any allocated block
    fn overlaps(&self, cidr: &Ipv4Net) -> bool;

    /// Get the number of available blocks for a given prefix length
    fn available_count(&self, prefix_len: u8) -> usize;

    /// Get all allocated CIDR blocks
    fn allocated_blocks(&self) -> Vec<Ipv4Net>;
}

/// Subnet allocator for a specific address space
///
/// Manages CIDR block allocation within a parent network (e.g., 10.100.0.0/12).
/// Uses first-fit allocation strategy with automatic gap detection.
#[derive(Debug, Clone)]
pub struct SubnetAllocator {
    /// Parent address space
    address_space: Ipv4Net,
    /// Allocated CIDR blocks, keyed by network address for ordering
    /// Key is the u32 representation of the network address
    allocated: BTreeMap<u32, Ipv4Net>,
}

impl SubnetAllocator {
    /// Create a new allocator for the given address space
    pub fn new(address_space: Ipv4Net) -> Self {
        Self {
            address_space,
            allocated: BTreeMap::new(),
        }
    }

    /// Create allocator for tenant subnets
    pub fn for_tenants() -> Self {
        Self::new(crate::address_space::TENANT_SPACE)
    }

    /// Create allocator for node type subnets
    pub fn for_node_types() -> Self {
        Self::new(crate::address_space::NODE_TYPE_SPACE)
    }

    /// Create allocator for geographic subnets
    pub fn for_geographic() -> Self {
        Self::new(crate::address_space::GEOGRAPHIC_SPACE)
    }

    /// Create allocator for resource pool subnets
    pub fn for_resource_pools() -> Self {
        Self::new(crate::address_space::RESOURCE_POOL_SPACE)
    }

    /// Get the parent address space
    pub fn address_space(&self) -> Ipv4Net {
        self.address_space
    }

    /// Convert an IPv4 address to u32 for ordering
    fn ip_to_u32(ip: Ipv4Addr) -> u32 {
        u32::from(ip)
    }

    /// Convert u32 back to IPv4 address
    fn u32_to_ip(val: u32) -> Ipv4Addr {
        Ipv4Addr::from(val)
    }

    /// Find the first available gap that can fit the requested prefix length
    fn find_gap(&self, prefix_len: u8) -> Option<Ipv4Net> {
        // Requested block size in addresses
        let block_size = 1u32 << (32 - prefix_len);

        // Start from the beginning of the address space
        let space_start = Self::ip_to_u32(self.address_space.network());
        let space_end = Self::ip_to_u32(self.address_space.broadcast());

        // Current position to search from
        let mut current = space_start;

        // Iterate through allocated blocks to find gaps
        for (&block_start, cidr) in &self.allocated {
            // Calculate block end
            let block_end = block_start + (1u32 << (32 - cidr.prefix_len())) - 1;

            // Check if there's a gap before this block
            if current < block_start {
                // Align current to block boundary
                let aligned = self.align_to_prefix(current, prefix_len);

                // Check if aligned block fits in the gap
                if aligned < block_start && aligned + block_size - 1 < block_start {
                    // Verify it's within address space
                    if aligned + block_size - 1 <= space_end {
                        return Ipv4Net::new(Self::u32_to_ip(aligned), prefix_len).ok();
                    }
                }
            }

            // Move current past this block
            current = current.max(block_end + 1);
        }

        // Check if there's space after the last block
        let aligned = self.align_to_prefix(current, prefix_len);
        if aligned + block_size - 1 <= space_end {
            return Ipv4Net::new(Self::u32_to_ip(aligned), prefix_len).ok();
        }

        None
    }

    /// Align an address to a prefix boundary
    fn align_to_prefix(&self, addr: u32, prefix_len: u8) -> u32 {
        let block_size = 1u32 << (32 - prefix_len);
        let mask = !(block_size - 1);
        let aligned = addr & mask;

        // If we're past the aligned address, move to next block
        if aligned < addr {
            aligned + block_size
        } else {
            aligned
        }
    }

    /// Check if two CIDR blocks overlap
    fn blocks_overlap(a: &Ipv4Net, b: &Ipv4Net) -> bool {
        a.contains(&b.network())
            || a.contains(&b.broadcast())
            || b.contains(&a.network())
            || b.contains(&a.broadcast())
    }
}

impl CidrAllocator for SubnetAllocator {
    fn allocate(&mut self, prefix_len: u8) -> Result<Ipv4Net> {
        // Validate prefix length
        if prefix_len < self.address_space.prefix_len() {
            return Err(Error::InvalidCidr(format!(
                "Prefix length {} is smaller than address space prefix {}",
                prefix_len,
                self.address_space.prefix_len()
            )));
        }

        if prefix_len > 30 {
            return Err(Error::InvalidCidr(
                "Prefix length cannot be greater than 30 (minimum 4 hosts)".to_string(),
            ));
        }

        // Find an available gap
        let cidr = self.find_gap(prefix_len).ok_or_else(|| {
            Error::CidrExhausted(format!(
                "No available /{} blocks in {}",
                prefix_len, self.address_space
            ))
        })?;

        // Insert into allocated set
        let key = Self::ip_to_u32(cidr.network());
        self.allocated.insert(key, cidr);

        Ok(cidr)
    }

    fn reserve(&mut self, cidr: Ipv4Net) -> Result<()> {
        // Verify CIDR is within address space
        if !self.address_space.contains(&cidr.network())
            || !self.address_space.contains(&cidr.broadcast())
        {
            return Err(Error::CidrOutOfRange(
                cidr.to_string(),
                self.address_space.to_string(),
            ));
        }

        // Check for overlaps with existing allocations
        for existing in self.allocated.values() {
            if Self::blocks_overlap(&cidr, existing) {
                return Err(Error::CidrOverlap(cidr.to_string(), existing.to_string()));
            }
        }

        // Insert into allocated set
        let key = Self::ip_to_u32(cidr.network());
        self.allocated.insert(key, cidr);

        Ok(())
    }

    fn release(&mut self, cidr: Ipv4Net) -> Result<()> {
        let key = Self::ip_to_u32(cidr.network());

        if self.allocated.remove(&key).is_none() {
            return Err(Error::InvalidCidr(format!(
                "CIDR {} was not allocated",
                cidr
            )));
        }

        Ok(())
    }

    fn is_allocated(&self, cidr: &Ipv4Net) -> bool {
        let key = Self::ip_to_u32(cidr.network());
        self.allocated.get(&key).map(|c| c == cidr).unwrap_or(false)
    }

    fn overlaps(&self, cidr: &Ipv4Net) -> bool {
        for existing in self.allocated.values() {
            if Self::blocks_overlap(cidr, existing) {
                return true;
            }
        }
        false
    }

    fn available_count(&self, prefix_len: u8) -> usize {
        if prefix_len < self.address_space.prefix_len() {
            return 0;
        }

        // Total possible blocks of this size in address space
        let total_bits = prefix_len - self.address_space.prefix_len();
        let total_blocks = 1usize << total_bits;

        // Count non-overlapping blocks that would fit
        // This is an approximation; exact count requires more complex calculation
        let mut count = 0;
        let mut test_allocator = self.clone();

        while test_allocator.allocate(prefix_len).is_ok() {
            count += 1;
            if count >= total_blocks {
                break;
            }
        }

        count
    }

    fn allocated_blocks(&self) -> Vec<Ipv4Net> {
        self.allocated.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_allocator_creation() {
        let space = Ipv4Net::from_str("10.100.0.0/12").unwrap();
        let allocator = SubnetAllocator::new(space);

        assert_eq!(allocator.address_space(), space);
        assert!(allocator.allocated_blocks().is_empty());
    }

    #[test]
    fn test_allocate_sequential() {
        let space = Ipv4Net::from_str("10.100.0.0/16").unwrap();
        let mut allocator = SubnetAllocator::new(space);

        // Allocate three /20 subnets
        let cidr1 = allocator.allocate(20).unwrap();
        let cidr2 = allocator.allocate(20).unwrap();
        let cidr3 = allocator.allocate(20).unwrap();

        // Should be sequential
        assert_eq!(cidr1.to_string(), "10.100.0.0/20");
        assert_eq!(cidr2.to_string(), "10.100.16.0/20");
        assert_eq!(cidr3.to_string(), "10.100.32.0/20");

        // All should be allocated
        assert!(allocator.is_allocated(&cidr1));
        assert!(allocator.is_allocated(&cidr2));
        assert!(allocator.is_allocated(&cidr3));
    }

    #[test]
    fn test_reserve_specific() {
        let space = Ipv4Net::from_str("10.100.0.0/16").unwrap();
        let mut allocator = SubnetAllocator::new(space);

        let specific = Ipv4Net::from_str("10.100.32.0/20").unwrap();
        allocator.reserve(specific).unwrap();

        assert!(allocator.is_allocated(&specific));

        // Next allocation should skip the reserved block
        let cidr1 = allocator.allocate(20).unwrap();
        assert_eq!(cidr1.to_string(), "10.100.0.0/20");

        let cidr2 = allocator.allocate(20).unwrap();
        assert_eq!(cidr2.to_string(), "10.100.16.0/20");

        let cidr3 = allocator.allocate(20).unwrap();
        // Should skip 10.100.32.0/20 (reserved)
        assert_eq!(cidr3.to_string(), "10.100.48.0/20");
    }

    #[test]
    fn test_release_and_reuse() {
        let space = Ipv4Net::from_str("10.100.0.0/16").unwrap();
        let mut allocator = SubnetAllocator::new(space);

        let cidr1 = allocator.allocate(20).unwrap();
        let cidr2 = allocator.allocate(20).unwrap();
        let _cidr3 = allocator.allocate(20).unwrap();

        // Release the second one
        allocator.release(cidr2).unwrap();
        assert!(!allocator.is_allocated(&cidr2));

        // Next allocation should reuse the released block
        let cidr4 = allocator.allocate(20).unwrap();
        assert_eq!(cidr4, cidr2);
    }

    #[test]
    fn test_overlap_detection() {
        let space = Ipv4Net::from_str("10.100.0.0/16").unwrap();
        let mut allocator = SubnetAllocator::new(space);

        // Allocate a /20
        let cidr = allocator.allocate(20).unwrap();

        // Try to reserve an overlapping /24
        let overlapping = Ipv4Net::from_str("10.100.8.0/24").unwrap();
        let result = allocator.reserve(overlapping);

        assert!(result.is_err());
        assert!(matches!(result, Err(Error::CidrOverlap(_, _))));
    }

    #[test]
    fn test_out_of_range() {
        let space = Ipv4Net::from_str("10.100.0.0/16").unwrap();
        let mut allocator = SubnetAllocator::new(space);

        // Try to reserve CIDR outside address space
        let outside = Ipv4Net::from_str("192.168.0.0/24").unwrap();
        let result = allocator.reserve(outside);

        assert!(result.is_err());
        assert!(matches!(result, Err(Error::CidrOutOfRange(_, _))));
    }

    #[test]
    fn test_exhaustion() {
        let space = Ipv4Net::from_str("10.100.0.0/20").unwrap();
        let mut allocator = SubnetAllocator::new(space);

        // Allocate all /24 blocks (16 blocks in a /20)
        for _ in 0..16 {
            allocator.allocate(24).unwrap();
        }

        // Next allocation should fail
        let result = allocator.allocate(24);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::CidrExhausted(_))));
    }

    #[test]
    fn test_prefix_too_small() {
        let space = Ipv4Net::from_str("10.100.0.0/16").unwrap();
        let mut allocator = SubnetAllocator::new(space);

        // Try to allocate /12 from /16 space
        let result = allocator.allocate(12);
        assert!(result.is_err());
    }

    #[test]
    fn test_factory_methods() {
        let tenant_alloc = SubnetAllocator::for_tenants();
        assert_eq!(
            tenant_alloc.address_space(),
            crate::address_space::TENANT_SPACE
        );

        let node_alloc = SubnetAllocator::for_node_types();
        assert_eq!(
            node_alloc.address_space(),
            crate::address_space::NODE_TYPE_SPACE
        );

        let geo_alloc = SubnetAllocator::for_geographic();
        assert_eq!(
            geo_alloc.address_space(),
            crate::address_space::GEOGRAPHIC_SPACE
        );

        let pool_alloc = SubnetAllocator::for_resource_pools();
        assert_eq!(
            pool_alloc.address_space(),
            crate::address_space::RESOURCE_POOL_SPACE
        );
    }

    #[test]
    fn test_overlaps_check() {
        let space = Ipv4Net::from_str("10.100.0.0/16").unwrap();
        let mut allocator = SubnetAllocator::new(space);

        let cidr = allocator.allocate(20).unwrap();

        // Check overlap with allocated block
        let overlapping = Ipv4Net::from_str("10.100.8.0/24").unwrap();
        assert!(allocator.overlaps(&overlapping));

        // Check non-overlapping
        let non_overlapping = Ipv4Net::from_str("10.100.128.0/24").unwrap();
        assert!(!allocator.overlaps(&non_overlapping));
    }
}
