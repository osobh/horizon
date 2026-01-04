use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: Uuid,
    pub action_description: String,
    pub estimated_cost: f64,
    pub risk_level: RiskLevel,
    pub status: ApprovalStatus,
    pub requested_at: chrono::DateTime<chrono::Utc>,
    pub resolved_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl ApprovalRequest {
    pub fn new(action_description: String, estimated_cost: f64, risk_level: RiskLevel) -> Self {
        Self {
            id: Uuid::new_v4(),
            action_description,
            estimated_cost,
            risk_level,
            status: ApprovalStatus::Pending,
            requested_at: chrono::Utc::now(),
            resolved_at: None,
        }
    }

    pub fn approve(&mut self) {
        self.status = ApprovalStatus::Approved;
        self.resolved_at = Some(chrono::Utc::now());
    }

    pub fn reject(&mut self) {
        self.status = ApprovalStatus::Rejected;
        self.resolved_at = Some(chrono::Utc::now());
    }

    pub fn expire(&mut self) {
        self.status = ApprovalStatus::Expired;
        self.resolved_at = Some(chrono::Utc::now());
    }

    pub fn is_pending(&self) -> bool {
        self.status == ApprovalStatus::Pending
    }

    pub fn is_approved(&self) -> bool {
        self.status == ApprovalStatus::Approved
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskLevel {
    pub fn from_cost(cost: f64) -> Self {
        if cost < 10.0 {
            RiskLevel::Low
        } else if cost < 50.0 {
            RiskLevel::Medium
        } else if cost < 100.0 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }
}

pub struct ApprovalGate {
    pending_requests: Vec<ApprovalRequest>,
    auto_approve_threshold: f64,
}

impl ApprovalGate {
    pub fn new(auto_approve_threshold: f64) -> Self {
        Self {
            pending_requests: Vec::new(),
            auto_approve_threshold,
        }
    }

    pub fn request_approval(
        &mut self,
        action_description: String,
        estimated_cost: f64,
    ) -> Result<Uuid> {
        let risk_level = RiskLevel::from_cost(estimated_cost);
        let mut request = ApprovalRequest::new(action_description, estimated_cost, risk_level);

        // Auto-approve if below threshold
        if estimated_cost <= self.auto_approve_threshold {
            request.approve();
        }

        let id = request.id;
        self.pending_requests.push(request);
        Ok(id)
    }

    pub fn approve(&mut self, request_id: Uuid) -> Result<()> {
        let request = self
            .pending_requests
            .iter_mut()
            .find(|r| r.id == request_id)
            .ok_or_else(|| AgentError::ApprovalRequired("Request not found".to_string()))?;

        if !request.is_pending() {
            return Err(AgentError::ApprovalRequired(format!(
                "Request already resolved with status: {:?}",
                request.status
            )));
        }

        request.approve();
        Ok(())
    }

    pub fn reject(&mut self, request_id: Uuid) -> Result<()> {
        let request = self
            .pending_requests
            .iter_mut()
            .find(|r| r.id == request_id)
            .ok_or_else(|| AgentError::ApprovalRequired("Request not found".to_string()))?;

        if !request.is_pending() {
            return Err(AgentError::ApprovalRequired(format!(
                "Request already resolved with status: {:?}",
                request.status
            )));
        }

        request.reject();
        Ok(())
    }

    pub fn is_approved(&self, request_id: Uuid) -> Result<bool> {
        let request = self
            .pending_requests
            .iter()
            .find(|r| r.id == request_id)
            .ok_or_else(|| AgentError::ApprovalRequired("Request not found".to_string()))?;

        Ok(request.is_approved())
    }

    pub fn get_pending_requests(&self) -> Vec<&ApprovalRequest> {
        self.pending_requests
            .iter()
            .filter(|r| r.is_pending())
            .collect()
    }

    pub fn cleanup_resolved(&mut self, max_age_hours: i64) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(max_age_hours);

        self.pending_requests
            .retain(|req| req.is_pending() || req.resolved_at.is_none_or(|t| t > cutoff));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approval_request_creation() {
        let req = ApprovalRequest::new("Test action".to_string(), 50.0, RiskLevel::Medium);

        assert_eq!(req.action_description, "Test action");
        assert_eq!(req.estimated_cost, 50.0);
        assert_eq!(req.risk_level, RiskLevel::Medium);
        assert_eq!(req.status, ApprovalStatus::Pending);
        assert!(req.resolved_at.is_none());
    }

    #[test]
    fn test_approval_request_approve() {
        let mut req = ApprovalRequest::new("Test action".to_string(), 50.0, RiskLevel::Medium);

        req.approve();
        assert_eq!(req.status, ApprovalStatus::Approved);
        assert!(req.resolved_at.is_some());
    }

    #[test]
    fn test_approval_request_reject() {
        let mut req = ApprovalRequest::new("Test action".to_string(), 50.0, RiskLevel::Medium);

        req.reject();
        assert_eq!(req.status, ApprovalStatus::Rejected);
        assert!(req.resolved_at.is_some());
    }

    #[test]
    fn test_approval_request_expire() {
        let mut req = ApprovalRequest::new("Test action".to_string(), 50.0, RiskLevel::Medium);

        req.expire();
        assert_eq!(req.status, ApprovalStatus::Expired);
        assert!(req.resolved_at.is_some());
    }

    #[test]
    fn test_approval_request_status_checks() {
        let mut req = ApprovalRequest::new("Test action".to_string(), 50.0, RiskLevel::Medium);

        assert!(req.is_pending());
        assert!(!req.is_approved());

        req.approve();
        assert!(!req.is_pending());
        assert!(req.is_approved());
    }

    #[test]
    fn test_risk_level_from_cost() {
        assert_eq!(RiskLevel::from_cost(5.0), RiskLevel::Low);
        assert_eq!(RiskLevel::from_cost(25.0), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_cost(75.0), RiskLevel::High);
        assert_eq!(RiskLevel::from_cost(150.0), RiskLevel::Critical);
    }

    #[test]
    fn test_approval_gate_creation() {
        let gate = ApprovalGate::new(10.0);
        assert_eq!(gate.auto_approve_threshold, 10.0);
        assert_eq!(gate.pending_requests.len(), 0);
    }

    #[test]
    fn test_approval_gate_auto_approve() {
        let mut gate = ApprovalGate::new(10.0);

        let id = gate
            .request_approval("Low cost action".to_string(), 5.0)
            .unwrap();
        assert!(gate.is_approved(id).unwrap());
    }

    #[test]
    fn test_approval_gate_requires_approval() {
        let mut gate = ApprovalGate::new(10.0);

        let id = gate
            .request_approval("High cost action".to_string(), 50.0)
            .unwrap();
        assert!(!gate.is_approved(id).unwrap());
    }

    #[test]
    fn test_approval_gate_approve() {
        let mut gate = ApprovalGate::new(10.0);

        let id = gate
            .request_approval("Test action".to_string(), 50.0)
            .unwrap();
        assert!(!gate.is_approved(id).unwrap());

        gate.approve(id).unwrap();
        assert!(gate.is_approved(id).unwrap());
    }

    #[test]
    fn test_approval_gate_reject() {
        let mut gate = ApprovalGate::new(10.0);

        let id = gate
            .request_approval("Test action".to_string(), 50.0)
            .unwrap();
        gate.reject(id).unwrap();

        assert!(!gate.is_approved(id).unwrap());
    }

    #[test]
    fn test_approval_gate_approve_nonexistent() {
        let mut gate = ApprovalGate::new(10.0);

        let result = gate.approve(Uuid::new_v4());
        assert!(result.is_err());
    }

    #[test]
    fn test_approval_gate_approve_already_resolved() {
        let mut gate = ApprovalGate::new(10.0);

        let id = gate
            .request_approval("Test action".to_string(), 50.0)
            .unwrap();
        gate.approve(id).unwrap();

        let result = gate.approve(id);
        assert!(result.is_err());
    }

    #[test]
    fn test_approval_gate_get_pending_requests() {
        let mut gate = ApprovalGate::new(10.0);

        let id1 = gate.request_approval("Action 1".to_string(), 50.0).unwrap();
        gate.request_approval("Action 2".to_string(), 60.0).unwrap();
        gate.request_approval("Action 3".to_string(), 5.0).unwrap(); // Auto-approved

        let pending = gate.get_pending_requests();
        assert_eq!(pending.len(), 2);

        gate.approve(id1).unwrap();
        let pending = gate.get_pending_requests();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_approval_gate_cleanup_resolved() {
        let mut gate = ApprovalGate::new(10.0);

        let id1 = gate.request_approval("Action 1".to_string(), 50.0).unwrap();
        let id2 = gate.request_approval("Action 2".to_string(), 60.0).unwrap();

        gate.approve(id1).unwrap();
        gate.reject(id2).unwrap();

        assert_eq!(gate.pending_requests.len(), 2);

        gate.cleanup_resolved(0);
        assert_eq!(gate.pending_requests.len(), 0);
    }

    #[test]
    fn test_approval_gate_cleanup_keeps_pending() {
        let mut gate = ApprovalGate::new(10.0);

        gate.request_approval("Pending action".to_string(), 50.0)
            .unwrap();
        let id2 = gate
            .request_approval("Resolved action".to_string(), 60.0)
            .unwrap();

        gate.approve(id2).unwrap();

        gate.cleanup_resolved(0);
        assert_eq!(gate.pending_requests.len(), 1);
        assert_eq!(gate.get_pending_requests().len(), 1);
    }
}
