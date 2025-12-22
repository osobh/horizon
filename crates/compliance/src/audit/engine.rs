//! Main audit log engine implementation

use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Duration, Utc};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

use crate::audit::{
    chain::{AuditChain, ChainState},
    compliance::{AuditComplianceReport, ComplianceIssue, ComplianceIssueType, ComplianceIssueSeverity, CoverageAssessment},
    entry::AuditLogEntry,
    integrity::{IntegrityError, IntegrityErrorType, IntegrityErrorSeverity, IntegrityVerificationResult},
    query::{AuditQuery, SortOrder},
    types::{AuditEventType, AuditOutcome, AuditSeverity},
};
use crate::data_classification::DataMetadata;
use crate::error::{ComplianceError, ComplianceResult};

/// Main audit log engine
#[derive(Debug)]
pub struct AuditLogEngine {
    /// Active audit chains
    chains: HashMap<Uuid, AuditChain>,
    /// All audit entries indexed by ID
    entries: HashMap<Uuid, AuditLogEntry>,
    /// Entries organized by chain
    entries_by_chain: HashMap<Uuid, Vec<Uuid>>,
    /// Index by event type
    event_type_index: HashMap<AuditEventType, Vec<Uuid>>,
    /// Index by actor
    actor_index: HashMap<String, Vec<Uuid>>,
    /// Index by target
    target_index: HashMap<String, Vec<Uuid>>,
    /// Time-based index
    time_index: HashMap<i64, Vec<Uuid>>,
}

impl AuditLogEngine {
    /// Create new audit log engine
    pub fn new() -> Self {
        Self {
            chains: HashMap::new(),
            entries: HashMap::new(),
            entries_by_chain: HashMap::new(),
            event_type_index: HashMap::new(),
            actor_index: HashMap::new(),
            target_index: HashMap::new(),
            time_index: HashMap::new(),
        }
    }

    /// Create new audit chain
    pub fn create_audit_chain(&mut self, max_entries: usize) -> ComplianceResult<Uuid> {
        let mut chain = AuditChain::new(max_entries);
        let chain_id = chain.chain_id;
        
        self.chains.insert(chain_id, chain);
        self.entries_by_chain.insert(chain_id, Vec::new());
        
        Ok(chain_id)
    }

    /// Add audit entry to a chain
    pub fn add_audit_entry(
        &mut self,
        chain_id: Uuid,
        event_type: AuditEventType,
        severity: AuditSeverity,
        outcome: AuditOutcome,
        actor: String,
        target: String,
        description: String,
        metadata: Option<DataMetadata>,
        additional_data: HashMap<String, String>,
    ) -> ComplianceResult<Uuid> {
        // Get the chain
        let chain = self.chains.get_mut(&chain_id).ok_or_else(|| {
            ComplianceError::AuditError(format!("Audit chain not found: {}", chain_id))
        })?;

        // Create the entry
        let mut entry = AuditLogEntry::new(
            event_type,
            severity,
            outcome,
            actor.clone(),
            target.clone(),
            description,
        );

        // Add metadata if provided
        if let Some(meta) = metadata {
            entry.data_categories = vec![meta.category];
            for (key, value) in meta.tags {
                entry.add_metadata(key, value);
            }
        }

        // Add additional data
        for (key, value) in additional_data {
            entry.add_metadata(key, value);
        }

        let entry_id = entry.id;

        // Add to chain
        chain.add_entry(entry.clone())?;

        // Index the entry
        self.entries.insert(entry_id, entry.clone());
        self.entries_by_chain
            .entry(chain_id)
            .or_insert_with(Vec::new)
            .push(entry_id);

        // Update indices
        self.event_type_index
            .entry(event_type)
            .or_insert_with(Vec::new)
            .push(entry_id);
        
        self.actor_index
            .entry(actor)
            .or_insert_with(Vec::new)
            .push(entry_id);
        
        self.target_index
            .entry(target)
            .or_insert_with(Vec::new)
            .push(entry_id);

        // Time index (by hour)
        let hour_key = entry.timestamp.timestamp() / 3600;
        self.time_index
            .entry(hour_key)
            .or_insert_with(Vec::new)
            .push(entry_id);

        Ok(entry_id)
    }

    /// Query audit entries
    pub fn query_entries(&self, query: &AuditQuery) -> Vec<AuditLogEntry> {
        let mut results = Vec::new();

        // Start with all entries or filtered by time
        let candidate_ids: Vec<Uuid> = if let (Some(start), Some(end)) = (query.start_time, query.end_time) {
            self.get_entries_in_time_range(start, end)
        } else {
            self.entries.keys().cloned().collect()
        };

        for entry_id in candidate_ids {
            if let Some(entry) = self.entries.get(&entry_id) {
                if self.entry_matches_query(entry, query) {
                    results.push(entry.clone());
                }
            }
        }

        // Apply sorting
        match query.sort_order {
            SortOrder::TimestampAsc => results.sort_by_key(|e| e.timestamp),
            SortOrder::TimestampDesc => results.sort_by_key(|e| std::cmp::Reverse(e.timestamp)),
            SortOrder::SeverityAsc => results.sort_by_key(|e| e.severity),
            SortOrder::SeverityDesc => results.sort_by_key(|e| std::cmp::Reverse(e.severity)),
        }

        // Apply pagination
        if let Some(offset) = query.offset {
            results = results.into_iter().skip(offset).collect();
        }

        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        results
    }

    /// Verify integrity of a chain
    pub fn verify_chain_integrity(&self, chain_id: Uuid) -> ComplianceResult<IntegrityVerificationResult> {
        let start = Instant::now();
        let mut result = IntegrityVerificationResult::new();

        let chain = self.chains.get(&chain_id).ok_or_else(|| {
            ComplianceError::AuditError(format!("Chain not found: {}", chain_id))
        })?;

        // Verify chain integrity
        if !chain.verify_integrity() {
            result.add_error(IntegrityError::new(
                chain_id,
                IntegrityErrorType::BrokenChain,
                "Chain integrity verification failed".to_string(),
                IntegrityErrorSeverity::Critical,
            ));
        }

        // Verify individual entries
        for entry in &chain.entries {
            if !entry.verify_integrity() {
                result.add_error(IntegrityError::new(
                    entry.id,
                    IntegrityErrorType::HashMismatch,
                    "Entry hash verification failed".to_string(),
                    IntegrityErrorSeverity::Severe,
                ));
            }
            result.entries_verified += 1;
        }

        result.set_statistics(
            chain.entries.len(),
            start.elapsed().as_millis() as u64,
        );

        Ok(result)
    }

    /// Generate compliance report
    pub fn generate_compliance_report(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        framework: String,
    ) -> ComplianceResult<AuditComplianceReport> {
        let mut report = AuditComplianceReport::new(start_time, end_time, framework.clone());

        // Analyze coverage
        let mut coverage = CoverageAssessment::new();
        let mut event_type_counts = HashMap::new();
        let mut severity_counts = HashMap::new();

        for entry in self.entries.values() {
            if entry.timestamp >= start_time && entry.timestamp <= end_time {
                *event_type_counts.entry(entry.event_type).or_insert(0) += 1;
                *severity_counts.entry(entry.severity).or_insert(0) += 1;
            }
        }

        // Determine covered event types
        coverage.event_types_covered = event_type_counts.keys().cloned().collect();
        coverage.severity_distribution = severity_counts.iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        coverage.calculate_coverage();

        report.coverage = coverage;

        // Check for compliance issues
        self.check_compliance_issues(&mut report, &framework);

        // Calculate metrics
        report.set_metric("total_entries".to_string(), self.entries.len() as f64);
        report.set_metric("total_chains".to_string(), self.chains.len() as f64);

        Ok(report)
    }

    // Helper methods

    fn get_entries_in_time_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<Uuid> {
        let start_hour = start.timestamp() / 3600;
        let end_hour = end.timestamp() / 3600;
        
        let mut result = Vec::new();
        for hour in start_hour..=end_hour {
            if let Some(entries) = self.time_index.get(&hour) {
                for entry_id in entries {
                    if let Some(entry) = self.entries.get(entry_id) {
                        if entry.timestamp >= start && entry.timestamp <= end {
                            result.push(*entry_id);
                        }
                    }
                }
            }
        }
        result
    }

    fn entry_matches_query(&self, entry: &AuditLogEntry, query: &AuditQuery) -> bool {
        // Check event type filter
        if !query.event_types.is_empty() && !query.event_types.contains(&entry.event_type) {
            return false;
        }

        // Check severity filter
        if !query.severity_levels.is_empty() && !query.severity_levels.contains(&entry.severity) {
            return false;
        }

        // Check outcome filter
        if !query.outcomes.is_empty() && !query.outcomes.contains(&entry.outcome) {
            return false;
        }

        // Check actor filter
        if let Some(ref actor) = query.actor {
            if !entry.actor.contains(actor) {
                return false;
            }
        }

        // Check target filter
        if let Some(ref target) = query.target {
            if !entry.target.contains(target) {
                return false;
            }
        }

        // Check description filter
        if let Some(ref desc) = query.description_contains {
            if !entry.description.contains(desc) {
                return false;
            }
        }

        // Check correlation ID
        if let Some(ref corr_id) = query.correlation_id {
            if entry.correlation_id != Some(*corr_id) {
                return false;
            }
        }

        true
    }

    fn check_compliance_issues(&self, report: &mut AuditComplianceReport, framework: &str) {
        // Framework-specific checks
        match framework {
            "GDPR" => self.check_gdpr_compliance(report),
            "HIPAA" => self.check_hipaa_compliance(report),
            "SOC2" => self.check_soc2_compliance(report),
            _ => {}
        }
    }

    fn check_gdpr_compliance(&self, report: &mut AuditComplianceReport) {
        // Check for data deletion audit trails
        let deletion_events = self.event_type_index.get(&AuditEventType::DataDeletion);
        if deletion_events.is_none() || deletion_events.unwrap().is_empty() {
            report.add_issue(ComplianceIssue {
                issue_id: Uuid::new_v4(),
                issue_type: ComplianceIssueType::MissingEvents,
                severity: ComplianceIssueSeverity::High,
                description: "No data deletion events found".to_string(),
                requirement: "GDPR Article 17 - Right to erasure".to_string(),
                occurrences: 0,
                example_entries: Vec::new(),
                remediation: "Ensure all data deletions are properly audited".to_string(),
            });
        }
    }

    fn check_hipaa_compliance(&self, report: &mut AuditComplianceReport) {
        // HIPAA-specific checks would go here
        report.add_recommendation("Ensure PHI access is properly audited".to_string());
    }

    fn check_soc2_compliance(&self, report: &mut AuditComplianceReport) {
        // SOC2-specific checks would go here
        report.add_recommendation("Verify all system changes are audited".to_string());
    }
}

impl Default for AuditLogEngine {
    fn default() -> Self {
        Self::new()
    }
}