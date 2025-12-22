use crate::db::PolicyRepository;
use crate::error::{GovernorErrorExt, HpcError, Result};
use crate::models::Policy;
use hpc_policy::parse_policy;

#[derive(Debug, Clone)]
pub struct PolicyService {
    repository: PolicyRepository,
}

impl PolicyService {
    pub fn new(repository: PolicyRepository) -> Self {
        Self { repository }
    }

    pub async fn create_policy(
        &self,
        name: &str,
        content: &str,
        description: Option<&str>,
        created_by: &str,
    ) -> Result<Policy> {
        self.validate_policy_content(content)?;

        self.repository
            .create(name, content, description, created_by)
            .await
    }

    pub async fn get_policy(&self, name: &str) -> Result<Policy> {
        self.repository.get_by_name(name).await
    }

    pub async fn list_policies(&self, enabled_only: bool) -> Result<Vec<Policy>> {
        self.repository.list(enabled_only).await
    }

    pub async fn update_policy(
        &self,
        name: &str,
        content: &str,
        description: Option<&str>,
        created_by: &str,
    ) -> Result<Policy> {
        self.validate_policy_content(content)?;

        self.repository
            .update(name, content, description, created_by)
            .await
    }

    pub async fn delete_policy(&self, name: &str) -> Result<()> {
        self.repository.delete(name).await
    }

    fn validate_policy_content(&self, content: &str) -> Result<()> {
        parse_policy(content).map_err(|e| {
            HpcError::invalid_policy_content(format!("Invalid policy YAML: {}", e))
        })?;

        Ok(())
    }
}
