//! Security management for package integrity and access control

use std::sync::Arc;
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use anyhow::{Result, anyhow};
use uuid::Uuid;
use sha2::{Sha256, Digest};
use dashmap::DashMap;

/// Security manager for package integrity and access control
#[derive(Debug)]
pub struct SecurityManager {
    trusted_clusters: Arc<RwLock<HashSet<String>>>,
    package_signatures: Arc<DashMap<Uuid, String>>,
    access_control_policies: Arc<DashMap<String, AccessPolicy>>,
    audit_log: Arc<RwLock<Vec<SecurityAuditEntry>>>,
}

#[derive(Debug, Clone)]
struct AccessPolicy {
    cluster_id: String,
    allowed_operations: Vec<String>,
    rate_limits: HashMap<String, u32>,
    security_clearance: u32,
}

#[derive(Debug, Clone)]
struct SecurityAuditEntry {
    timestamp: u64,
    cluster_id: String,
    operation: String,
    package_id: Option<Uuid>,
    success: bool,
    details: String,
}

impl SecurityManager {
    pub fn new() -> Self {
        Self {
            trusted_clusters: Arc::new(RwLock::new(HashSet::new())),
            package_signatures: Arc::new(DashMap::new()),
            access_control_policies: Arc::new(DashMap::new()),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn add_trusted_cluster(&self, cluster_id: String) -> Result<()> {
        let mut trusted = self.trusted_clusters.write().await;
        trusted.insert(cluster_id.clone());
        
        self.log_audit_entry(
            cluster_id,
            "add_trusted_cluster".to_string(),
            None,
            true,
            "Cluster added to trusted list".to_string(),
        ).await;
        
        Ok(())
    }
    
    pub async fn is_trusted(&self, cluster_id: &str) -> bool {
        let trusted = self.trusted_clusters.read().await;
        trusted.contains(cluster_id)
    }
    
    pub async fn verify_package_signature(&self, package_id: Uuid, signature: &str) -> Result<bool> {
        if let Some(expected_sig) = self.package_signatures.get(&package_id) {
            Ok(expected_sig.value() == signature)
        } else {
            Err(anyhow!("No signature found for package {}", package_id))
        }
    }

    pub async fn sign_package(&self, package_id: Uuid, content: &[u8]) -> Result<String> {
        // Calculate SHA256 hash
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash = hasher.finalize();
        let signature = format!("{:x}", hash);

        self.package_signatures.insert(package_id, signature.clone());

        Ok(signature)
    }
    
    pub async fn check_access(&self, cluster_id: &str, operation: &str) -> Result<bool> {
        if let Some(policy) = self.access_control_policies.get(cluster_id) {
            let allowed = policy.allowed_operations.contains(&operation.to_string());

            self.log_audit_entry(
                cluster_id.to_string(),
                operation.to_string(),
                None,
                allowed,
                format!("Access check for operation: {}", operation),
            ).await;

            Ok(allowed)
        } else {
            // No policy means no access
            self.log_audit_entry(
                cluster_id.to_string(),
                operation.to_string(),
                None,
                false,
                "No access policy found".to_string(),
            ).await;

            Ok(false)
        }
    }
    
    pub async fn set_access_policy(
        &self,
        cluster_id: String,
        allowed_operations: Vec<String>,
        security_clearance: u32,
    ) -> Result<()> {
        use std::collections::HashMap;
        let policy = AccessPolicy {
            cluster_id: cluster_id.clone(),
            allowed_operations,
            rate_limits: HashMap::new(),
            security_clearance,
        };

        self.access_control_policies.insert(cluster_id.clone(), policy);

        self.log_audit_entry(
            cluster_id,
            "set_access_policy".to_string(),
            None,
            true,
            format!("Security clearance level: {}", security_clearance),
        ).await;

        Ok(())
    }
    
    async fn log_audit_entry(
        &self,
        cluster_id: String,
        operation: String,
        package_id: Option<Uuid>,
        success: bool,
        details: String,
    ) {
        let entry = SecurityAuditEntry {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            cluster_id,
            operation,
            package_id,
            success,
            details,
        };
        
        let mut log = self.audit_log.write().await;
        log.push(entry);
        
        // Keep only last 10000 entries
        if log.len() > 10000 {
            log.drain(0..log.len() - 10000);
        }
    }
    
    pub async fn get_audit_log(&self, limit: usize) -> Vec<SecurityAuditEntry> {
        let log = self.audit_log.read().await;
        log.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}