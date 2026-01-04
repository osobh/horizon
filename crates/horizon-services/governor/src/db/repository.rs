use crate::error::{GovernorErrorExt, HpcError, Result};
use crate::models::{Policy, PolicyVersion};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct PolicyRepository {
    pool: PgPool,
}

impl PolicyRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create(
        &self,
        name: &str,
        content: &str,
        description: Option<&str>,
        created_by: &str,
    ) -> Result<Policy> {
        let policy = sqlx::query_as::<_, Policy>(
            r#"
            INSERT INTO policies (name, content, description, created_by)
            VALUES ($1, $2, $3, $4)
            RETURNING *
            "#,
        )
        .bind(name)
        .bind(content)
        .bind(description)
        .bind(created_by)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| {
            if let sqlx::Error::Database(ref db_err) = e {
                if db_err.is_unique_violation() {
                    return HpcError::policy_already_exists(name);
                }
            }
            HpcError::from(e)
        })?;

        self.create_version(policy.id, policy.version, content, created_by)
            .await?;

        Ok(policy)
    }

    pub async fn get_by_name(&self, name: &str) -> Result<Policy> {
        sqlx::query_as::<_, Policy>(
            r#"
            SELECT * FROM policies
            WHERE name = $1
            "#,
        )
        .bind(name)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::policy_not_found(name))
    }

    pub async fn get_by_id(&self, id: Uuid) -> Result<Policy> {
        sqlx::query_as::<_, Policy>(
            r#"
            SELECT * FROM policies
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::policy_not_found(id.to_string()))
    }

    pub async fn list(&self, enabled_only: bool) -> Result<Vec<Policy>> {
        let query = if enabled_only {
            "SELECT * FROM policies WHERE enabled = true ORDER BY created_at DESC"
        } else {
            "SELECT * FROM policies ORDER BY created_at DESC"
        };

        let policies = sqlx::query_as::<_, Policy>(query)
            .fetch_all(&self.pool)
            .await?;

        Ok(policies)
    }

    pub async fn update(
        &self,
        name: &str,
        content: &str,
        description: Option<&str>,
        created_by: &str,
    ) -> Result<Policy> {
        let existing = self.get_by_name(name).await?;

        let new_version = existing.version + 1;

        let policy = sqlx::query_as::<_, Policy>(
            r#"
            UPDATE policies
            SET content = $1, description = $2, version = $3, updated_at = NOW()
            WHERE name = $4
            RETURNING *
            "#,
        )
        .bind(content)
        .bind(description)
        .bind(new_version)
        .bind(name)
        .fetch_one(&self.pool)
        .await?;

        self.create_version(policy.id, new_version, content, created_by)
            .await?;

        Ok(policy)
    }

    pub async fn delete(&self, name: &str) -> Result<()> {
        let result = sqlx::query(
            r#"
            DELETE FROM policies
            WHERE name = $1
            "#,
        )
        .bind(name)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(HpcError::policy_not_found(name));
        }

        Ok(())
    }

    pub async fn get_versions(&self, name: &str) -> Result<Vec<PolicyVersion>> {
        let policy = self.get_by_name(name).await?;

        let versions = sqlx::query_as::<_, PolicyVersion>(
            r#"
            SELECT * FROM policy_versions
            WHERE policy_id = $1
            ORDER BY version DESC
            "#,
        )
        .bind(policy.id)
        .fetch_all(&self.pool)
        .await?;

        Ok(versions)
    }

    async fn create_version(
        &self,
        policy_id: Uuid,
        version: i32,
        content: &str,
        created_by: &str,
    ) -> Result<PolicyVersion> {
        let version_record = sqlx::query_as::<_, PolicyVersion>(
            r#"
            INSERT INTO policy_versions (policy_id, version, content, created_by)
            VALUES ($1, $2, $3, $4)
            RETURNING *
            "#,
        )
        .bind(policy_id)
        .bind(version)
        .bind(content)
        .bind(created_by)
        .fetch_one(&self.pool)
        .await?;

        Ok(version_record)
    }

    pub async fn get_all_enabled_policies(&self) -> Result<Vec<Policy>> {
        let policies = sqlx::query_as::<_, Policy>(
            r#"
            SELECT * FROM policies
            WHERE enabled = true
            ORDER BY created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(policies)
    }
}
