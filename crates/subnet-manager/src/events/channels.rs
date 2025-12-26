//! Well-known channel IDs for subnet events
//!
//! These channel IDs are used for broadcasting subnet-related events
//! across the HPC platform via hpc-channels.

use uuid::Uuid;

/// A channel identifier for subnet events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(Uuid);

impl ChannelId {
    /// Create a new channel ID from a static string
    pub const fn from_static(name: &'static str) -> Self {
        // Use a deterministic UUID based on the channel name
        // This ensures the same name always produces the same ID
        Self(Uuid::from_bytes(Self::hash_name(name)))
    }

    /// Hash a channel name to 16 bytes for UUID
    const fn hash_name(name: &'static str) -> [u8; 16] {
        let bytes = name.as_bytes();
        let mut result = [0u8; 16];

        // Simple deterministic hash for const context
        let mut i = 0;
        while i < bytes.len() && i < 16 {
            result[i] = bytes[i];
            i += 1;
        }

        // XOR fold for longer names
        while i < bytes.len() {
            result[i % 16] ^= bytes[i];
            i += 1;
        }

        result
    }

    /// Get the underlying UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }

    /// Get the channel name (for display purposes)
    pub fn name(&self) -> &'static str {
        // Look up known channels
        if *self == SUBNET_LIFECYCLE {
            "subnet.lifecycle"
        } else if *self == SUBNET_TOPOLOGY {
            "subnet.topology"
        } else if *self == SUBNET_ASSIGNMENTS {
            "subnet.assignments"
        } else if *self == SUBNET_ROUTES {
            "subnet.routes"
        } else if *self == SUBNET_WIREGUARD {
            "subnet.wireguard"
        } else if *self == SUBNET_POLICIES {
            "subnet.policies"
        } else {
            "unknown"
        }
    }
}

impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// Well-known Subnet Channel IDs
// ============================================================================

/// Channel for subnet lifecycle events (created, deleted, status changes)
pub const SUBNET_LIFECYCLE: ChannelId = ChannelId::from_static("subnet.lifecycle");

/// Channel for topology snapshot broadcasts
pub const SUBNET_TOPOLOGY: ChannelId = ChannelId::from_static("subnet.topology");

/// Channel for node assignment events
pub const SUBNET_ASSIGNMENTS: ChannelId = ChannelId::from_static("subnet.assignments");

/// Channel for cross-subnet route events
pub const SUBNET_ROUTES: ChannelId = ChannelId::from_static("subnet.routes");

/// Channel for WireGuard configuration events
pub const SUBNET_WIREGUARD: ChannelId = ChannelId::from_static("subnet.wireguard");

/// Channel for policy events (created, updated, deleted)
pub const SUBNET_POLICIES: ChannelId = ChannelId::from_static("subnet.policies");

/// Get all subnet-related channel IDs
pub fn all_subnet_channels() -> &'static [ChannelId] {
    &[
        SUBNET_LIFECYCLE,
        SUBNET_TOPOLOGY,
        SUBNET_ASSIGNMENTS,
        SUBNET_ROUTES,
        SUBNET_WIREGUARD,
        SUBNET_POLICIES,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_id_deterministic() {
        // Same name should always produce same ID
        let id1 = ChannelId::from_static("subnet.lifecycle");
        let id2 = ChannelId::from_static("subnet.lifecycle");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_channel_id_different_names() {
        // Different names should produce different IDs
        let id1 = ChannelId::from_static("subnet.lifecycle");
        let id2 = ChannelId::from_static("subnet.topology");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_channel_name_lookup() {
        assert_eq!(SUBNET_LIFECYCLE.name(), "subnet.lifecycle");
        assert_eq!(SUBNET_TOPOLOGY.name(), "subnet.topology");
        assert_eq!(SUBNET_ASSIGNMENTS.name(), "subnet.assignments");
    }

    #[test]
    fn test_all_channels() {
        let channels = all_subnet_channels();
        assert_eq!(channels.len(), 6);
    }
}
