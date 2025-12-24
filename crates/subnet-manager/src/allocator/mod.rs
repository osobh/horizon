//! IP and CIDR allocation modules
//!
//! Provides efficient allocation of:
//! - CIDR blocks from address spaces (subnet allocation)
//! - IP addresses within subnets (host allocation)

mod ip_allocator;
mod subnet_allocator;

pub use ip_allocator::{IpAllocator, SubnetIpAllocator};
pub use subnet_allocator::{CidrAllocator, SubnetAllocator};
