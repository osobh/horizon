//! Per-subnet IP address allocator
//!
//! Allocates individual IP addresses within a subnet CIDR block.
//! Uses a BTreeSet for O(log n) operations and efficient gap detection.

use crate::{Error, Result};
use ipnet::Ipv4Net;
use std::collections::BTreeSet;
use std::net::Ipv4Addr;

/// Trait for IP address allocation within a subnet
pub trait IpAllocator: Send + Sync {
    /// Allocate the next available IP address
    fn allocate(&mut self) -> Option<Ipv4Addr>;

    /// Reserve a specific IP address
    fn reserve(&mut self, ip: Ipv4Addr) -> Result<()>;

    /// Release a previously allocated IP address
    fn release(&mut self, ip: Ipv4Addr);

    /// Check if an IP address is currently allocated
    fn is_allocated(&self, ip: Ipv4Addr) -> bool;

    /// Get the number of available IP addresses
    fn available_count(&self) -> usize;

    /// Get all allocated IP addresses
    fn allocated_ips(&self) -> Vec<Ipv4Addr>;
}

/// IP allocator for a specific subnet
///
/// Manages IP allocation within a subnet CIDR block.
/// Reserves network address, gateway (first host), and broadcast address.
#[derive(Debug, Clone)]
pub struct SubnetIpAllocator {
    /// Subnet CIDR block
    cidr: Ipv4Net,
    /// Set of allocated IP addresses (stored as u32 for efficient ordering)
    allocated: BTreeSet<u32>,
    /// First allocatable host address
    first_host: u32,
    /// Last allocatable host address
    last_host: u32,
    /// Hint for next allocation (optimization)
    next_hint: u32,
    /// Reserved IPs (gateway, etc.)
    reserved: BTreeSet<u32>,
}

impl SubnetIpAllocator {
    /// Create a new IP allocator for the given subnet
    ///
    /// Automatically reserves:
    /// - Network address (first IP)
    /// - Gateway address (second IP, first host)
    /// - Broadcast address (last IP)
    pub fn new(cidr: Ipv4Net) -> Self {
        let network = u32::from(cidr.network());
        let broadcast = u32::from(cidr.broadcast());

        // For /31 or /32, there are no usable hosts
        let (first_host, last_host) = if cidr.prefix_len() >= 31 {
            (network, network) // No usable hosts
        } else {
            // First usable host is after network address
            // Last usable host is before broadcast
            (network + 1, broadcast - 1)
        };

        let mut reserved = BTreeSet::new();
        reserved.insert(network); // Network address
        reserved.insert(broadcast); // Broadcast address

        // Reserve gateway (first host) unless it's the same as network
        if first_host != network {
            reserved.insert(first_host);
        }

        Self {
            cidr,
            allocated: BTreeSet::new(),
            first_host: if cidr.prefix_len() >= 31 { network } else { first_host + 1 }, // Start after gateway
            last_host,
            next_hint: if cidr.prefix_len() >= 31 { network } else { first_host + 1 },
            reserved,
        }
    }

    /// Create a new allocator without gateway reservation
    ///
    /// Only reserves network and broadcast addresses.
    pub fn new_no_gateway(cidr: Ipv4Net) -> Self {
        let network = u32::from(cidr.network());
        let broadcast = u32::from(cidr.broadcast());

        let (first_host, last_host) = if cidr.prefix_len() >= 31 {
            (network, network)
        } else {
            (network + 1, broadcast - 1)
        };

        let mut reserved = BTreeSet::new();
        reserved.insert(network);
        reserved.insert(broadcast);

        Self {
            cidr,
            allocated: BTreeSet::new(),
            first_host,
            last_host,
            next_hint: first_host,
            reserved,
        }
    }

    /// Get the subnet CIDR
    pub fn cidr(&self) -> Ipv4Net {
        self.cidr
    }

    /// Get the gateway IP (first host)
    pub fn gateway(&self) -> Ipv4Addr {
        Ipv4Addr::from(u32::from(self.cidr.network()) + 1)
    }

    /// Check if an IP is within this subnet
    pub fn contains(&self, ip: Ipv4Addr) -> bool {
        self.cidr.contains(&ip)
    }

    /// Check if an IP is reserved (network, broadcast, gateway)
    pub fn is_reserved(&self, ip: Ipv4Addr) -> bool {
        self.reserved.contains(&u32::from(ip))
    }

    /// Find the next available IP starting from the hint
    fn find_next_available(&self) -> Option<u32> {
        // Try from hint to last_host
        for ip in self.next_hint..=self.last_host {
            if !self.allocated.contains(&ip) && !self.reserved.contains(&ip) {
                return Some(ip);
            }
        }

        // Wrap around and try from first_host to hint
        for ip in self.first_host..self.next_hint {
            if !self.allocated.contains(&ip) && !self.reserved.contains(&ip) {
                return Some(ip);
            }
        }

        None
    }

    /// Restore allocator state from persisted allocations
    pub fn with_allocated(mut self, allocated: &[Ipv4Addr]) -> Self {
        for ip in allocated {
            let ip_u32 = u32::from(*ip);
            if ip_u32 >= self.first_host && ip_u32 <= self.last_host {
                self.allocated.insert(ip_u32);
            }
        }
        self
    }
}

impl IpAllocator for SubnetIpAllocator {
    fn allocate(&mut self) -> Option<Ipv4Addr> {
        let ip = self.find_next_available()?;
        self.allocated.insert(ip);

        // Update hint for next allocation
        self.next_hint = if ip < self.last_host {
            ip + 1
        } else {
            self.first_host
        };

        Some(Ipv4Addr::from(ip))
    }

    fn reserve(&mut self, ip: Ipv4Addr) -> Result<()> {
        let ip_u32 = u32::from(ip);

        // Check if IP is within subnet
        if !self.contains(ip) {
            return Err(Error::IpNotInSubnet(ip, uuid::Uuid::nil()));
        }

        // Check if IP is reserved
        if self.reserved.contains(&ip_u32) {
            return Err(Error::IpAlreadyAllocated(ip));
        }

        // Check if already allocated
        if self.allocated.contains(&ip_u32) {
            return Err(Error::IpAlreadyAllocated(ip));
        }

        self.allocated.insert(ip_u32);
        Ok(())
    }

    fn release(&mut self, ip: Ipv4Addr) {
        let ip_u32 = u32::from(ip);

        // Don't allow releasing reserved addresses
        if self.reserved.contains(&ip_u32) {
            return;
        }

        // Remove from allocated set
        if self.allocated.remove(&ip_u32) {
            // Update hint for better reuse
            if ip_u32 < self.next_hint {
                self.next_hint = ip_u32;
            }
        }
    }

    fn is_allocated(&self, ip: Ipv4Addr) -> bool {
        let ip_u32 = u32::from(ip);
        self.allocated.contains(&ip_u32) || self.reserved.contains(&ip_u32)
    }

    fn available_count(&self) -> usize {
        if self.last_host < self.first_host {
            return 0;
        }

        let total = (self.last_host - self.first_host + 1) as usize;
        let allocated = self.allocated.len();

        // Count reserved IPs in the allocatable range
        let reserved_in_range = self.reserved.iter()
            .filter(|&&ip| ip >= self.first_host && ip <= self.last_host)
            .count();

        total.saturating_sub(allocated + reserved_in_range)
    }

    fn allocated_ips(&self) -> Vec<Ipv4Addr> {
        self.allocated.iter().map(|&ip| Ipv4Addr::from(ip)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_allocator_creation() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let allocator = SubnetIpAllocator::new(cidr);

        assert_eq!(allocator.cidr(), cidr);
        // /24 = 256 IPs - network - broadcast - gateway = 253 usable
        assert_eq!(allocator.available_count(), 253);
    }

    #[test]
    fn test_gateway_reservation() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let allocator = SubnetIpAllocator::new(cidr);

        // Gateway (10.100.0.1) should be reserved
        assert!(allocator.is_reserved(Ipv4Addr::new(10, 100, 0, 1)));

        // First allocatable should be 10.100.0.2
        let mut allocator = allocator;
        let ip = allocator.allocate().unwrap();
        assert_eq!(ip, Ipv4Addr::new(10, 100, 0, 2));
    }

    #[test]
    fn test_no_gateway_mode() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut allocator = SubnetIpAllocator::new_no_gateway(cidr);

        // First allocation should be 10.100.0.1
        let ip = allocator.allocate().unwrap();
        assert_eq!(ip, Ipv4Addr::new(10, 100, 0, 1));
    }

    #[test]
    fn test_sequential_allocation() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        let ip1 = allocator.allocate().unwrap();
        let ip2 = allocator.allocate().unwrap();
        let ip3 = allocator.allocate().unwrap();

        assert_eq!(ip1, Ipv4Addr::new(10, 100, 0, 2));
        assert_eq!(ip2, Ipv4Addr::new(10, 100, 0, 3));
        assert_eq!(ip3, Ipv4Addr::new(10, 100, 0, 4));
    }

    #[test]
    fn test_release_and_reuse() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        let ip1 = allocator.allocate().unwrap();
        let ip2 = allocator.allocate().unwrap();
        let ip3 = allocator.allocate().unwrap();

        // Release the second one
        allocator.release(ip2);
        assert!(!allocator.is_allocated(ip2));

        // Next allocation should reuse the released IP
        let ip4 = allocator.allocate().unwrap();
        assert_eq!(ip4, ip2);
    }

    #[test]
    fn test_reserve_specific() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        // Reserve a specific IP
        let specific = Ipv4Addr::new(10, 100, 0, 100);
        allocator.reserve(specific).unwrap();
        assert!(allocator.is_allocated(specific));

        // Sequential allocation should skip the reserved IP
        for _ in 0..97 {
            // Allocate 2-99 (98 IPs, but 10.100.0.100 is reserved)
            let ip = allocator.allocate().unwrap();
            assert_ne!(ip, specific);
        }
    }

    #[test]
    fn test_reserve_already_allocated() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        let ip = allocator.allocate().unwrap();

        // Try to reserve an already allocated IP
        let result = allocator.reserve(ip);
        assert!(result.is_err());
    }

    #[test]
    fn test_reserve_out_of_subnet() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        // Try to reserve an IP outside the subnet
        let outside = Ipv4Addr::new(192, 168, 0, 1);
        let result = allocator.reserve(outside);
        assert!(result.is_err());
    }

    #[test]
    fn test_exhaustion() {
        let cidr = Ipv4Net::from_str("10.100.0.0/28").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        // /28 = 16 IPs - network - broadcast - gateway = 13 usable
        let mut allocated = Vec::new();
        while let Some(ip) = allocator.allocate() {
            allocated.push(ip);
        }

        assert_eq!(allocated.len(), 13);
        assert_eq!(allocator.available_count(), 0);

        // Next allocation should return None
        assert!(allocator.allocate().is_none());
    }

    #[test]
    fn test_contains() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let allocator = SubnetIpAllocator::new(cidr);

        assert!(allocator.contains(Ipv4Addr::new(10, 100, 0, 50)));
        assert!(!allocator.contains(Ipv4Addr::new(10, 100, 1, 50)));
        assert!(!allocator.contains(Ipv4Addr::new(192, 168, 0, 1)));
    }

    #[test]
    fn test_with_allocated() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let existing = vec![
            Ipv4Addr::new(10, 100, 0, 10),
            Ipv4Addr::new(10, 100, 0, 20),
            Ipv4Addr::new(10, 100, 0, 30),
        ];

        let allocator = SubnetIpAllocator::new(cidr).with_allocated(&existing);

        // Existing IPs should be allocated
        assert!(allocator.is_allocated(Ipv4Addr::new(10, 100, 0, 10)));
        assert!(allocator.is_allocated(Ipv4Addr::new(10, 100, 0, 20)));
        assert!(allocator.is_allocated(Ipv4Addr::new(10, 100, 0, 30)));

        // Available count should reflect the pre-allocated IPs
        assert_eq!(allocator.available_count(), 250); // 253 - 3
    }

    #[test]
    fn test_allocated_ips() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        allocator.allocate().unwrap();
        allocator.allocate().unwrap();
        allocator.allocate().unwrap();

        let ips = allocator.allocated_ips();
        assert_eq!(ips.len(), 3);
        assert!(ips.contains(&Ipv4Addr::new(10, 100, 0, 2)));
        assert!(ips.contains(&Ipv4Addr::new(10, 100, 0, 3)));
        assert!(ips.contains(&Ipv4Addr::new(10, 100, 0, 4)));
    }

    #[test]
    fn test_small_subnet() {
        // /30 = 4 IPs: network, 2 hosts, broadcast
        let cidr = Ipv4Net::from_str("10.100.0.0/30").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        // With gateway reserved, only 1 usable IP
        assert_eq!(allocator.available_count(), 1);

        let ip = allocator.allocate().unwrap();
        assert_eq!(ip, Ipv4Addr::new(10, 100, 0, 2));

        // Should be exhausted
        assert!(allocator.allocate().is_none());
    }

    #[test]
    fn test_release_reserved_is_noop() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut allocator = SubnetIpAllocator::new(cidr);

        let gateway = Ipv4Addr::new(10, 100, 0, 1);
        let count_before = allocator.available_count();

        // Try to release reserved gateway
        allocator.release(gateway);

        // Should still be reserved
        assert!(allocator.is_reserved(gateway));
        assert_eq!(allocator.available_count(), count_before);
    }
}
