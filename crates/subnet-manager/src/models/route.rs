//! Cross-subnet routing models

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Direction of cross-subnet routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteDirection {
    /// Traffic flows in both directions
    Bidirectional,
    /// Traffic flows only from source to destination
    Unidirectional,
}

/// Status of a cross-subnet route
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteStatus {
    /// Route is active and enabled
    Active,
    /// Route is disabled
    Disabled,
    /// Route has expired
    Expired,
    /// Route is pending approval
    Pending,
}

impl RouteStatus {
    /// Derive status from a route
    pub fn from_route(route: &CrossSubnetRoute) -> Self {
        if !route.enabled {
            return RouteStatus::Disabled;
        }
        if let Some(expires_at) = route.expires_at {
            if Utc::now() > expires_at {
                return RouteStatus::Expired;
            }
        }
        RouteStatus::Active
    }
}

/// Port range for route restrictions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortRange {
    /// Start port (inclusive)
    pub start: u16,
    /// End port (inclusive)
    pub end: u16,
}

impl PortRange {
    /// Create a single port range
    pub fn single(port: u16) -> Self {
        Self {
            start: port,
            end: port,
        }
    }

    /// Create a port range
    pub fn range(start: u16, end: u16) -> Self {
        Self { start, end }
    }

    /// Check if a port is within this range
    pub fn contains(&self, port: u16) -> bool {
        port >= self.start && port <= self.end
    }

    /// Common port ranges
    pub fn ssh() -> Self {
        Self::single(22)
    }

    pub fn http() -> Self {
        Self::single(80)
    }

    pub fn https() -> Self {
        Self::single(443)
    }

    pub fn wireguard_default() -> Self {
        Self::single(51820)
    }

    pub fn high_ports() -> Self {
        Self::range(1024, 65535)
    }
}

/// Cross-subnet route configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSubnetRoute {
    /// Unique route identifier
    pub id: Uuid,
    /// Source subnet ID
    pub source_subnet_id: Uuid,
    /// Destination subnet ID
    pub destination_subnet_id: Uuid,
    /// Routing direction
    pub direction: RouteDirection,
    /// Allowed ports (None = all ports)
    pub allowed_ports: Option<Vec<PortRange>>,
    /// Allowed protocols (None = all protocols)
    /// Common values: "tcp", "udp", "icmp"
    pub allowed_protocols: Option<Vec<String>>,
    /// Whether route is currently enabled
    pub enabled: bool,
    /// Description/reason for route
    pub description: Option<String>,
    /// User who approved this route
    pub approved_by: Option<Uuid>,
    /// Approval timestamp
    pub approved_at: Option<DateTime<Utc>>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Expiration time (for temporary routes)
    pub expires_at: Option<DateTime<Utc>>,
    /// Priority (for conflict resolution)
    pub priority: i32,
    /// Custom metadata
    pub metadata: Option<serde_json::Value>,
}

impl CrossSubnetRoute {
    /// Create a new bidirectional route between two subnets
    pub fn new(source_subnet_id: Uuid, destination_subnet_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_subnet_id,
            destination_subnet_id,
            direction: RouteDirection::Bidirectional,
            allowed_ports: None,
            allowed_protocols: None,
            enabled: true,
            description: None,
            approved_by: None,
            approved_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
            priority: 0,
            metadata: None,
        }
    }

    /// Create a unidirectional route
    pub fn unidirectional(source_subnet_id: Uuid, destination_subnet_id: Uuid) -> Self {
        let mut route = Self::new(source_subnet_id, destination_subnet_id);
        route.direction = RouteDirection::Unidirectional;
        route
    }

    /// Set allowed ports
    pub fn with_ports(mut self, ports: Vec<PortRange>) -> Self {
        self.allowed_ports = Some(ports);
        self
    }

    /// Set allowed protocols
    pub fn with_protocols(mut self, protocols: Vec<String>) -> Self {
        self.allowed_protocols = Some(protocols);
        self
    }

    /// Set expiration
    pub fn with_expiration(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Check if route is currently active
    pub fn is_active(&self) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(expires_at) = self.expires_at {
            if Utc::now() > expires_at {
                return false;
            }
        }

        true
    }

    /// Check if traffic on a port/protocol is allowed
    pub fn allows_traffic(&self, port: u16, protocol: &str) -> bool {
        if !self.is_active() {
            return false;
        }

        // Check protocol
        if let Some(ref protocols) = self.allowed_protocols {
            if !protocols.iter().any(|p| p.eq_ignore_ascii_case(protocol)) {
                return false;
            }
        }

        // Check port
        if let Some(ref ports) = self.allowed_ports {
            if !ports.iter().any(|range| range.contains(port)) {
                return false;
            }
        }

        true
    }

    /// Get the reverse route ID for bidirectional routes
    pub fn reverse_pair(&self) -> (Uuid, Uuid) {
        (self.destination_subnet_id, self.source_subnet_id)
    }
}

/// Request to create a cross-subnet route
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRouteRequest {
    pub source_subnet_id: Uuid,
    pub destination_subnet_id: Uuid,
    pub direction: RouteDirection,
    pub allowed_ports: Option<Vec<PortRange>>,
    pub allowed_protocols: Option<Vec<String>>,
    pub description: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub priority: Option<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_range() {
        let range = PortRange::range(80, 443);
        assert!(range.contains(80));
        assert!(range.contains(443));
        assert!(range.contains(200));
        assert!(!range.contains(79));
        assert!(!range.contains(444));
    }

    #[test]
    fn test_route_creation() {
        let source = Uuid::new_v4();
        let dest = Uuid::new_v4();

        let route = CrossSubnetRoute::new(source, dest)
            .with_ports(vec![PortRange::ssh(), PortRange::https()])
            .with_protocols(vec!["tcp".to_string()]);

        assert!(route.is_active());
        assert!(route.allows_traffic(22, "tcp"));
        assert!(route.allows_traffic(443, "TCP")); // case insensitive
        assert!(!route.allows_traffic(80, "tcp")); // port not in list
        assert!(!route.allows_traffic(22, "udp")); // protocol not in list
    }

    #[test]
    fn test_route_expiration() {
        let source = Uuid::new_v4();
        let dest = Uuid::new_v4();

        let mut route = CrossSubnetRoute::new(source, dest);
        assert!(route.is_active());

        // Set expiration in the past
        route.expires_at = Some(Utc::now() - chrono::Duration::hours(1));
        assert!(!route.is_active());
    }

    #[test]
    fn test_unidirectional_route() {
        let source = Uuid::new_v4();
        let dest = Uuid::new_v4();

        let route = CrossSubnetRoute::unidirectional(source, dest);
        assert_eq!(route.direction, RouteDirection::Unidirectional);
    }
}
