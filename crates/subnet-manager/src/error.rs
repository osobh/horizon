//! Error types for subnet management

use std::net::Ipv4Addr;
use thiserror::Error;
use uuid::Uuid;

/// Result type for subnet operations
pub type Result<T> = std::result::Result<T, Error>;

/// Subnet manager errors
#[derive(Debug, Clone, Error)]
pub enum Error {
    // Subnet errors
    #[error("Subnet not found: {0}")]
    SubnetNotFound(Uuid),

    #[error("Subnet already exists: {0}")]
    SubnetAlreadyExists(String),

    #[error("Subnet is not empty, contains {0} nodes")]
    SubnetNotEmpty(usize),

    #[error("Subnet is in invalid state for operation: {state:?}")]
    InvalidSubnetState { state: String },

    // CIDR errors
    #[error("CIDR exhausted in address space: {0}")]
    CidrExhausted(String),

    #[error("Invalid CIDR: {0}")]
    InvalidCidr(String),

    #[error("CIDR overlap detected: {0} overlaps with {1}")]
    CidrOverlap(String, String),

    #[error("CIDR {0} is not within address space {1}")]
    CidrOutOfRange(String, String),

    // IP allocation errors
    #[error("No available IPs in subnet {0}")]
    NoAvailableIps(Uuid),

    #[error("IP {0} is already allocated")]
    IpAlreadyAllocated(Ipv4Addr),

    #[error("IP {0} is not allocated")]
    IpNotAllocated(Ipv4Addr),

    #[error("IP {0} is not in subnet {1}")]
    IpNotInSubnet(Ipv4Addr, Uuid),

    // Policy errors
    #[error("Policy not found: {0}")]
    PolicyNotFound(Uuid),

    #[error("No matching policy for node {0}")]
    NoMatchingPolicy(Uuid),

    #[error("Policy evaluation failed: {0}")]
    PolicyEvaluationFailed(String),

    // Assignment errors
    #[error("Node {0} is already assigned to subnet {1}")]
    NodeAlreadyAssigned(Uuid, Uuid),

    #[error("Node {0} is not assigned to any subnet")]
    NodeNotAssigned(Uuid),

    // Route errors
    #[error("Route not found: {0} -> {1}")]
    RouteNotFound(Uuid, Uuid),

    #[error("Route already exists: {0} -> {1}")]
    RouteAlreadyExists(Uuid, Uuid),

    #[error("Cannot create route to same subnet")]
    SelfRoute,

    // Migration errors
    #[error("Migration not found: {0}")]
    MigrationNotFound(Uuid),

    #[error("Migration failed: {0}")]
    MigrationFailed(String),

    #[error("Migration in progress for node {0}")]
    MigrationInProgress(Uuid),

    // Template errors
    #[error("Template not found: {0}")]
    TemplateNotFound(Uuid),

    // WireGuard errors
    #[error("WireGuard configuration error: {0}")]
    WireGuardConfig(String),

    #[error("WireGuard sync failed: {0}")]
    WireGuardSync(String),

    // Database errors
    #[error("Database error: {0}")]
    Database(String),

    // General errors
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Operation not permitted: {0}")]
    NotPermitted(String),
}

impl From<ipnet::PrefixLenError> for Error {
    fn from(e: ipnet::PrefixLenError) -> Self {
        Error::InvalidCidr(e.to_string())
    }
}

impl From<std::net::AddrParseError> for Error {
    fn from(e: std::net::AddrParseError) -> Self {
        Error::InvalidCidr(e.to_string())
    }
}

#[cfg(feature = "postgres")]
impl From<sqlx::Error> for Error {
    fn from(e: sqlx::Error) -> Self {
        Error::Database(e.to_string())
    }
}
