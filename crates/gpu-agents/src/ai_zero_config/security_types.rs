//! Security-related types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;
use std::time::Duration;

/// Security configuration
#[derive(Debug, Clone)]
pub struct SecurityConfiguration {
    pub encryption_config: EncryptionConfiguration,
    pub authentication_config: AuthenticationConfiguration,
    pub authorization_config: AuthorizationConfiguration,
    pub secrets_management: SecretsManagementConfiguration,
    pub network_security: NetworkSecurityConfiguration,
    pub compliance_settings: ComplianceConfiguration,
}

/// Encryption configuration
#[derive(Debug, Clone)]
pub struct EncryptionConfiguration {
    pub encryption_at_rest: bool,
    pub encryption_in_transit: bool,
    pub key_management: KeyManagementConfiguration,
    pub cipher_suites: Vec<String>,
    pub tls_version: TLSVersion,
}

/// Key management configuration
#[derive(Debug, Clone)]
pub struct KeyManagementConfiguration {
    pub provider: KeyManagementProvider,
    pub key_rotation_enabled: bool,
    pub key_rotation_interval: Duration,
    pub hsm_enabled: bool,
}

/// Key management providers
#[derive(Debug, Clone, PartialEq)]
pub enum KeyManagementProvider {
    CloudKMS,
    HashiCorpVault,
    AWSSecretsManager,
    AzureKeyVault,
    SelfManaged,
}

/// TLS versions
#[derive(Debug, Clone, PartialEq)]
pub enum TLSVersion {
    TLS10,
    TLS11,
    TLS12,
    TLS13,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthenticationConfiguration {
    pub primary_method: super::AuthenticationMethod,
    pub fallback_methods: Vec<super::AuthenticationMethod>,
    pub multi_factor_enabled: bool,
    pub session_management: SessionManagementConfiguration,
    pub oauth_providers: Vec<OAuthProvider>,
}

/// Session management configuration
#[derive(Debug, Clone)]
pub struct SessionManagementConfiguration {
    pub session_timeout: Duration,
    pub idle_timeout: Duration,
    pub concurrent_sessions_limit: Option<u32>,
    pub session_storage: SessionStorageType,
}

/// Session storage types
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStorageType {
    Memory,
    Database,
    Redis,
    Cookie,
    JWT,
}

/// OAuth providers
#[derive(Debug, Clone)]
pub struct OAuthProvider {
    pub provider_name: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    pub redirect_uri: String,
}

/// Authorization configuration
#[derive(Debug, Clone)]
pub struct AuthorizationConfiguration {
    pub authorization_model: AuthorizationModel,
    pub roles: Vec<Role>,
    pub permissions: Vec<Permission>,
    pub resource_policies: Vec<ResourcePolicy>,
}

/// Authorization models
#[derive(Debug, Clone, PartialEq)]
pub enum AuthorizationModel {
    RBAC,
    ABAC,
    ReBAC,
    ACL,
}

/// Roles definition
#[derive(Debug, Clone)]
pub struct Role {
    pub name: String,
    pub description: String,
    pub permissions: Vec<String>,
    pub inherits_from: Vec<String>,
}

/// Permissions definition
#[derive(Debug, Clone)]
pub struct Permission {
    pub name: String,
    pub description: String,
    pub resource_type: String,
    pub actions: Vec<String>,
}

/// Resource policies
#[derive(Debug, Clone)]
pub struct ResourcePolicy {
    pub resource_type: String,
    pub resource_id: Option<String>,
    pub policy_document: String,
}

/// Secrets management configuration
#[derive(Debug, Clone)]
pub struct SecretsManagementConfiguration {
    pub secrets_provider: SecretsProvider,
    pub automatic_rotation: bool,
    pub rotation_schedule: Option<String>,
    pub encryption_key: String,
    pub access_policies: Vec<SecretsAccessPolicy>,
}

/// Secrets providers
#[derive(Debug, Clone, PartialEq)]
pub enum SecretsProvider {
    Kubernetes,
    HashiCorpVault,
    AWSSecretsManager,
    AzureKeyVault,
    GCPSecretManager,
    External,
}

/// Secrets access policies
#[derive(Debug, Clone)]
pub struct SecretsAccessPolicy {
    pub secret_name: String,
    pub allowed_principals: Vec<String>,
    pub allowed_actions: Vec<String>,
    pub conditions: Vec<String>,
}

/// Network security configuration
#[derive(Debug, Clone)]
pub struct NetworkSecurityConfiguration {
    pub network_policies: Vec<NetworkPolicy>,
    pub firewall_rules: Vec<FirewallRule>,
    pub ddos_protection: bool,
    pub waf_enabled: bool,
    pub vpc_configuration: Option<VPCConfiguration>,
}

/// Network policies
#[derive(Debug, Clone)]
pub struct NetworkPolicy {
    pub name: String,
    pub namespace: String,
    pub pod_selector: HashMap<String, String>,
    pub ingress_rules: Vec<NetworkPolicyRule>,
    pub egress_rules: Vec<NetworkPolicyRule>,
}

/// Network policy rules
#[derive(Debug, Clone)]
pub struct NetworkPolicyRule {
    pub ports: Vec<NetworkPort>,
    pub from_selectors: Vec<NetworkPolicyPeer>,
    pub to_selectors: Vec<NetworkPolicyPeer>,
}

/// Network ports
#[derive(Debug, Clone)]
pub struct NetworkPort {
    pub port: u16,
    pub protocol: NetworkProtocol,
}

/// Network protocols
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkProtocol {
    TCP,
    UDP,
    SCTP,
}

/// Network policy peers
#[derive(Debug, Clone)]
pub struct NetworkPolicyPeer {
    pub pod_selector: Option<HashMap<String, String>>,
    pub namespace_selector: Option<HashMap<String, String>>,
    pub ip_block: Option<IPBlock>,
}

/// IP block definition
#[derive(Debug, Clone)]
pub struct IPBlock {
    pub cidr: String,
    pub except: Vec<String>,
}

/// Firewall rules
#[derive(Debug, Clone)]
pub struct FirewallRule {
    pub name: String,
    pub direction: FirewallDirection,
    pub action: FirewallAction,
    pub source_ranges: Vec<String>,
    pub destination_ranges: Vec<String>,
    pub ports: Vec<NetworkPort>,
    pub priority: u32,
}

/// Firewall directions
#[derive(Debug, Clone, PartialEq)]
pub enum FirewallDirection {
    Ingress,
    Egress,
}

/// Firewall actions
#[derive(Debug, Clone, PartialEq)]
pub enum FirewallAction {
    Allow,
    Deny,
    Log,
}

/// VPC configuration
#[derive(Debug, Clone)]
pub struct VPCConfiguration {
    pub vpc_id: String,
    pub subnets: Vec<SubnetConfiguration>,
    pub route_tables: Vec<RouteTable>,
    pub nat_gateways: Vec<NATGateway>,
    pub vpn_connections: Vec<VPNConnection>,
}

/// Subnet configuration
#[derive(Debug, Clone)]
pub struct SubnetConfiguration {
    pub subnet_id: String,
    pub cidr_block: String,
    pub availability_zone: String,
    pub subnet_type: SubnetType,
}

/// Subnet types
#[derive(Debug, Clone, PartialEq)]
pub enum SubnetType {
    Public,
    Private,
    Database,
    Cache,
}

/// Route tables
#[derive(Debug, Clone)]
pub struct RouteTable {
    pub route_table_id: String,
    pub routes: Vec<Route>,
    pub associated_subnets: Vec<String>,
}

/// Network routes
#[derive(Debug, Clone)]
pub struct Route {
    pub destination_cidr: String,
    pub target: RouteTarget,
}

/// Route targets
#[derive(Debug, Clone, PartialEq)]
pub enum RouteTarget {
    InternetGateway,
    NATGateway,
    VPCPeering,
    TransitGateway,
    VPNGateway,
    NetworkInterface,
}

/// NAT gateway configuration
#[derive(Debug, Clone)]
pub struct NATGateway {
    pub nat_gateway_id: String,
    pub subnet_id: String,
    pub elastic_ip: String,
}

/// VPN connection configuration
#[derive(Debug, Clone)]
pub struct VPNConnection {
    pub vpn_connection_id: String,
    pub customer_gateway_ip: String,
    pub vpn_type: VPNType,
    pub routing_type: VPNRoutingType,
}

/// VPN types
#[derive(Debug, Clone, PartialEq)]
pub enum VPNType {
    SiteToSite,
    ClientVPN,
    PointToPoint,
}

/// VPN routing types
#[derive(Debug, Clone, PartialEq)]
pub enum VPNRoutingType {
    Static,
    Dynamic,
}

/// Compliance configuration
#[derive(Debug, Clone)]
pub struct ComplianceConfiguration {
    pub frameworks: Vec<super::ComplianceFramework>,
    pub audit_logging: AuditLoggingConfiguration,
    pub data_residency: DataResidencyConfiguration,
    pub privacy_controls: PrivacyControlsConfiguration,
}

/// Audit logging configuration
#[derive(Debug, Clone)]
pub struct AuditLoggingConfiguration {
    pub enabled: bool,
    pub log_level: LogLevel,
    pub retention_days: u32,
    pub storage_location: String,
    pub encryption_enabled: bool,
}

/// Log levels
#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
    Critical,
}

/// Data residency configuration
#[derive(Debug, Clone)]
pub struct DataResidencyConfiguration {
    pub allowed_regions: Vec<String>,
    pub data_classification: HashMap<String, DataClassification>,
    pub cross_border_restrictions: Vec<CrossBorderRestriction>,
}

/// Data classification levels
#[derive(Debug, Clone, PartialEq)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
    PersonalData,
    SensitivePersonalData,
}

/// Cross-border data transfer restrictions
#[derive(Debug, Clone)]
pub struct CrossBorderRestriction {
    pub from_region: String,
    pub to_region: String,
    pub restriction_type: RestrictionType,
    pub exemptions: Vec<String>,
}

/// Restriction types
#[derive(Debug, Clone, PartialEq)]
pub enum RestrictionType {
    Prohibited,
    ConditionalAllowed,
    RequiresApproval,
    Allowed,
}

/// Privacy controls configuration
#[derive(Debug, Clone)]
pub struct PrivacyControlsConfiguration {
    pub data_minimization: bool,
    pub purpose_limitation: bool,
    pub consent_management: ConsentManagementConfiguration,
    pub right_to_erasure: bool,
    pub data_portability: bool,
}

/// Consent management configuration
#[derive(Debug, Clone)]
pub struct ConsentManagementConfiguration {
    pub enabled: bool,
    pub consent_types: Vec<ConsentType>,
    pub storage_duration: Duration,
    pub withdrawal_mechanism: WithdrawalMechanism,
}

/// Consent types
#[derive(Debug, Clone, PartialEq)]
pub enum ConsentType {
    Marketing,
    Analytics,
    Functional,
    Essential,
    ThirdParty,
}

/// Consent withdrawal mechanisms
#[derive(Debug, Clone, PartialEq)]
pub enum WithdrawalMechanism {
    UserInterface,
    API,
    Email,
    Phone,
    Mail,
}