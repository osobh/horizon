//! Network-related types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;
use std::time::Duration;

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfiguration {
    pub service_discovery: ServiceDiscoveryConfiguration,
    pub load_balancing: super::LoadBalancerConfiguration,
    pub ingress: super::IngressConfiguration,
    pub egress: EgressConfiguration,
    pub dns_configuration: DNSConfiguration,
}

/// Service discovery configuration
#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfiguration {
    pub enabled: bool,
    pub discovery_method: ServiceDiscoveryMethod,
    pub health_checking: bool,
    pub registration_ttl: Duration,
    pub custom_attributes: HashMap<String, String>,
}

/// Service discovery methods
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceDiscoveryMethod {
    DNS,
    Consul,
    Eureka,
    Kubernetes,
    Zookeeper,
    Custom,
}

/// Egress configuration
#[derive(Debug, Clone)]
pub struct EgressConfiguration {
    pub external_services: Vec<ExternalService>,
    pub proxy_configuration: Option<ProxyConfiguration>,
    pub ssl_configuration: SSLConfiguration,
}

/// External service configuration
#[derive(Debug, Clone)]
pub struct ExternalService {
    pub name: String,
    pub endpoints: Vec<String>,
    pub connection_timeout: Duration,
    pub read_timeout: Duration,
    pub retry_policy: Option<super::RetryPolicy>,
}

/// Proxy configuration
#[derive(Debug, Clone)]
pub struct ProxyConfiguration {
    pub proxy_type: ProxyType,
    pub proxy_host: String,
    pub proxy_port: u16,
    pub authentication: Option<ProxyAuthentication>,
}

/// Proxy types
#[derive(Debug, Clone, PartialEq)]
pub enum ProxyType {
    HTTP,
    HTTPS,
    SOCKS4,
    SOCKS5,
}

/// Proxy authentication
#[derive(Debug, Clone)]
pub struct ProxyAuthentication {
    pub username: String,
    pub password: String,
}

/// SSL configuration
#[derive(Debug, Clone)]
pub struct SSLConfiguration {
    pub ssl_enabled: bool,
    pub ssl_verification: SSLVerification,
    pub custom_ca_certificates: Vec<String>,
    pub client_certificates: Vec<ClientCertificate>,
}

/// SSL verification levels
#[derive(Debug, Clone, PartialEq)]
pub enum SSLVerification {
    Full,
    HostnameOnly,
    None,
}

/// Client certificates
#[derive(Debug, Clone)]
pub struct ClientCertificate {
    pub certificate: String,
    pub private_key: String,
    pub ca_certificate: Option<String>,
}

/// DNS configuration
#[derive(Debug, Clone)]
pub struct DNSConfiguration {
    pub dns_servers: Vec<String>,
    pub search_domains: Vec<String>,
    pub dns_policy: DNSPolicy,
    pub dns_caching: bool,
    pub dns_timeout: Duration,
}

/// DNS policies
#[derive(Debug, Clone, PartialEq)]
pub enum DNSPolicy {
    ClusterFirst,
    ClusterFirstWithHostNet,
    Default,
    None,
}