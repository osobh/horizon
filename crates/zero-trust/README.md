# ExoRust Zero-Trust Security Module

A comprehensive zero-trust security framework implementing modern security principles for the ExoRust system.

## Overview

This module provides enterprise-grade security capabilities following zero-trust principles:
- **Never trust, always verify**
- **Least privilege access**
- **Assume breach mentality**
- **Continuous verification**
- **Risk-based access control**
- **Defense in depth**

## Core Components

### 1. Identity Management (`identity.rs`)
- Multi-factor authentication (MFA) with configurable requirements
- Identity lifecycle management
- Continuous authentication with session monitoring
- Support for multiple authentication factors:
  - Password-based authentication
  - TOTP (Time-based One-Time Password)
  - Hardware tokens
  - Biometric authentication
  - SMS and Email-based verification
- Failed attempt tracking and automatic account lockout
- Cryptographic identity verification using SHA256

**Key Features:**
- Configurable MFA requirements (None, Optional, Required, Required with Hardware)
- Session-based authentication with token management
- Identity state management (Active, Suspended, Locked, Pending, Disabled)
- Concurrent authentication support

### 2. Device Trust (`device_trust.rs`)
- Hardware and software attestation
- Device trust scoring based on multiple factors
- Certificate-based device authentication
- Continuous compliance monitoring
- Device health assessment
- Support for various device types (Desktop, Laptop, Mobile, Server, IoT, Virtual)

**Key Features:**
- TPM-based attestation support
- X.509 certificate validation
- Trust score calculation with configurable thresholds
- Device state management (Trusted, Pending, Quarantined, Blocked)
- Compliance policy enforcement

### 3. Network Policy (`network_policy.rs`)
- Network microsegmentation
- Policy-based access control
- Traffic inspection and analysis
- Dynamic policy enforcement
- Support for multiple protocols (TCP, UDP, ICMP, HTTP/S, SSH, RDP, DNS)

**Key Features:**
- Segment-based network isolation
- Time-based and conditional policies
- Traffic anomaly detection
- Policy priority and cascading
- Connection state tracking

### 4. Behavior Analysis (`behavior_analysis.rs`)
- User and Entity Behavior Analytics (UEBA)
- Machine learning-based anomaly detection
- Pattern recognition and profiling
- Adaptive baseline adjustment
- Real-time threat detection

**Key Features:**
- Temporal and geographic anomaly detection
- Peer group comparison
- Behavioral pattern recognition
- Risk indicator tracking
- Adaptive learning capabilities

### 5. Risk Engine (`risk_engine.rs`)
- Multi-factor risk scoring
- Real-time threat assessment
- Contextual risk evaluation
- Adaptive risk-based access control
- Decision automation

**Key Features:**
- Configurable risk aggregation methods
- Policy-based risk decisions
- Historical risk tracking
- Contextual adjustments
- Risk trend analysis

### 6. Session Manager (`session_manager.rs`)
- Secure session lifecycle management
- Token-based authentication with rotation
- Continuous session verification
- Session risk assessment
- Concurrent session management

**Key Features:**
- Configurable session timeouts
- Session binding to device/location
- Token rotation and refresh
- Activity tracking
- Permission-based access control

### 7. Attestation (`attestation.rs`)
- Hardware-based attestation (TPM, Secure Enclave)
- Software integrity measurement
- Secure boot chain validation
- Remote attestation protocols
- Continuous integrity monitoring

**Key Features:**
- Multiple attestation types (TPM, Secure Enclave, UEFI, Software)
- Measurement validation against references
- Policy-based attestation requirements
- Certificate chain validation
- Integrity monitoring

## Security Features

### Cryptographic Operations
- SHA256/384/512 hashing
- HMAC-based token signing
- Secure random number generation
- Base64 encoding for tokens

### Error Handling
All modules use a comprehensive error type system (`ZeroTrustError`) with specific error variants for each security domain.

### Async/Await Support
All operations are async-ready for high-performance concurrent operations.

## Configuration

Each module has its own configuration structure with sensible defaults:

```rust
// Example: Identity Configuration
let config = IdentityConfig {
    max_auth_attempts: 3,
    token_validity_duration: Duration::hours(1),
    mfa_requirement: MfaRequirement::Required,
    session_timeout: Duration::hours(8),
    continuous_auth_enabled: true,
    continuous_auth_interval: Duration::minutes(5),
    verification_level: VerificationLevel::High,
};
```

## Testing

The module includes comprehensive test coverage with 10-12 test cases per component:
- Unit tests for all public APIs
- Edge case testing
- Security scenario testing
- Concurrent operation testing
- Performance benchmarking hooks

## Benchmarks

Performance benchmarks are available for all major operations:
```bash
cargo bench -p exorust-zero-trust
```

## Usage Example

```rust
use exorust_zero_trust::{
    identity::{IdentityProvider, IdentityConfig, AuthFactor},
    device_trust::{DeviceTrustManager, DeviceTrustConfig},
    risk_engine::{RiskEngine, RiskEngineConfig},
};

// Initialize providers
let identity_provider = IdentityProvider::new(IdentityConfig::default())?;
let device_manager = DeviceTrustManager::new(DeviceTrustConfig::default())?;
let risk_engine = RiskEngine::new(RiskEngineConfig::default())?;

// Create and authenticate user
let identity = identity_provider.create_identity("user@example.com".to_string(), "user@example.com".to_string()).await?;
let session = identity_provider.authenticate(
    identity.id,
    vec![
        AuthFactor::Password("secure_password".to_string()),
        AuthFactor::Totp("123456".to_string()),
    ]
).await?;
```

## Security Considerations

1. **Secrets Management**: All sensitive data (passwords, tokens) are hashed or encrypted
2. **Session Security**: Sessions have configurable timeouts and continuous verification
3. **Device Trust**: Devices must pass attestation and maintain trust scores
4. **Network Isolation**: Traffic is controlled by microsegmentation policies
5. **Behavioral Monitoring**: Continuous analysis detects anomalies
6. **Risk-Based Decisions**: All access decisions consider current risk levels

## License

This module is part of the ExoRust project and follows the same licensing terms.