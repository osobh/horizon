//! Token management commands for node onboarding.
//!
//! This module provides CLI commands for generating and managing join tokens
//! that enable curl-based node bootstrap:
//!
//! ```bash
//! stratoswarm token generate --ttl 24h --uses 10
//! ```

use crate::{config::CliConfig, output, Result};
use chrono::{Duration, Utc};
use clap::{Args, Subcommand};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Args)]
pub struct TokenArgs {
    #[command(subcommand)]
    pub command: TokenCommands,
}

#[derive(Debug, Clone, Subcommand)]
pub enum TokenCommands {
    /// Generate a new join token
    Generate(GenerateArgs),

    /// List active tokens
    List(ListArgs),

    /// Revoke a token
    Revoke(RevokeArgs),
}

#[derive(Debug, Clone, Args)]
pub struct GenerateArgs {
    /// Time-to-live for the token (e.g., "24h", "7d", "1h30m")
    #[arg(long, default_value = "24h")]
    pub ttl: String,

    /// Maximum number of uses (0 = unlimited)
    #[arg(long, default_value = "1")]
    pub uses: u32,

    /// Capabilities to grant to nodes using this token
    #[arg(long)]
    pub capabilities: Vec<String>,

    /// Optional description for the token
    #[arg(long)]
    pub description: Option<String>,

    /// Cluster host (uses configured default if not specified)
    #[arg(long)]
    pub cluster_host: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub struct ListArgs {
    /// Show expired tokens
    #[arg(long)]
    pub include_expired: bool,

    /// Output format (table, json)
    #[arg(long, default_value = "table")]
    pub format: String,
}

#[derive(Debug, Clone, Args)]
pub struct RevokeArgs {
    /// Token ID to revoke
    pub token_id: String,
}

/// Represents a generated join token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinToken {
    pub id: Uuid,
    pub token: String,
    pub created_at: chrono::DateTime<Utc>,
    pub expires_at: chrono::DateTime<Utc>,
    pub max_uses: u32,
    pub current_uses: u32,
    pub capabilities: Vec<String>,
    pub description: Option<String>,
}

impl JoinToken {
    /// Generate a new join token with the specified parameters.
    pub fn generate(ttl: Duration, max_uses: u32, capabilities: Vec<String>) -> Self {
        let now = Utc::now();
        let expires_at = now + ttl;

        // Generate a secure random token
        let token_bytes: [u8; 32] = rand::random();
        let token = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            token_bytes,
        );

        Self {
            id: Uuid::new_v4(),
            token,
            created_at: now,
            expires_at,
            max_uses,
            current_uses: 0,
            capabilities,
            description: None,
        }
    }

    /// Set a description for the token.
    #[must_use]
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Check if the token has expired.
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Check if the token can still be used.
    pub fn is_valid(&self) -> bool {
        !self.is_expired() && (self.max_uses == 0 || self.current_uses < self.max_uses)
    }

    /// Get the install URL for this token.
    pub fn install_url(&self, cluster_host: &str) -> String {
        format!(
            "https://{}/api/v1/install?token={}",
            cluster_host, self.token
        )
    }
}

/// API request to store a token
#[derive(Debug, Serialize)]
struct StoreTokenRequest {
    token: JoinToken,
}

/// API response for token operations
#[derive(Debug, Deserialize)]
struct TokenResponse {
    success: bool,
    #[serde(default)]
    message: Option<String>,
}

/// API response for listing tokens
#[derive(Debug, Deserialize)]
struct ListTokensResponse {
    tokens: Vec<JoinToken>,
}

/// Client for token API operations
struct TokenClient {
    client: Client,
    base_url: String,
    auth_token: Option<String>,
}

impl TokenClient {
    fn new(config: &CliConfig) -> Self {
        Self {
            client: Client::new(),
            base_url: config.api_endpoint.clone(),
            auth_token: config.auth_token.clone(),
        }
    }

    async fn store_token(&self, token: &JoinToken) -> Result<()> {
        let url = format!("{}/api/v1/tokens", self.base_url);

        let mut request = self.client.post(&url).json(&StoreTokenRequest {
            token: token.clone(),
        });

        if let Some(ref auth) = self.auth_token {
            request = request.header("Authorization", format!("Bearer {}", auth));
        }

        let response = request.send().await.map_err(|e| {
            crate::CliError::Command(format!("Failed to connect to cluster: {}", e))
        })?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(crate::CliError::Command(format!(
                "Failed to store token ({}): {}",
                status, body
            )))
        }
    }

    async fn list_tokens(&self, include_expired: bool) -> Result<Vec<JoinToken>> {
        let url = format!(
            "{}/api/v1/tokens?include_expired={}",
            self.base_url, include_expired
        );

        let mut request = self.client.get(&url);

        if let Some(ref auth) = self.auth_token {
            request = request.header("Authorization", format!("Bearer {}", auth));
        }

        let response = request.send().await.map_err(|e| {
            crate::CliError::Command(format!("Failed to connect to cluster: {}", e))
        })?;

        if response.status().is_success() {
            let list_response: ListTokensResponse = response.json().await.map_err(|e| {
                crate::CliError::Command(format!("Invalid response from cluster: {}", e))
            })?;
            Ok(list_response.tokens)
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(crate::CliError::Command(format!(
                "Failed to list tokens ({}): {}",
                status, body
            )))
        }
    }

    async fn revoke_token(&self, token_id: &str) -> Result<()> {
        let url = format!("{}/api/v1/tokens/{}", self.base_url, token_id);

        let mut request = self.client.delete(&url);

        if let Some(ref auth) = self.auth_token {
            request = request.header("Authorization", format!("Bearer {}", auth));
        }

        let response = request.send().await.map_err(|e| {
            crate::CliError::Command(format!("Failed to connect to cluster: {}", e))
        })?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(crate::CliError::Command(format!(
                "Failed to revoke token ({}): {}",
                status, body
            )))
        }
    }
}

/// Parse a duration string like "24h", "7d", "1h30m" into a chrono Duration.
fn parse_duration(s: &str) -> Result<Duration> {
    let mut total_seconds: i64 = 0;
    let mut current_num = String::new();

    for c in s.chars() {
        if c.is_ascii_digit() {
            current_num.push(c);
        } else {
            let num: i64 = current_num
                .parse()
                .map_err(|_| crate::CliError::Command(format!("Invalid duration: {}", s)))?;
            current_num.clear();

            match c {
                'd' => total_seconds += num * 86400,
                'h' => total_seconds += num * 3600,
                'm' => total_seconds += num * 60,
                's' => total_seconds += num,
                _ => {
                    return Err(crate::CliError::Command(format!(
                        "Invalid duration unit: {}",
                        c
                    )))
                }
            }
        }
    }

    // Handle trailing number without unit (assume seconds)
    if !current_num.is_empty() {
        let num: i64 = current_num
            .parse()
            .map_err(|_| crate::CliError::Command(format!("Invalid duration: {}", s)))?;
        total_seconds += num;
    }

    Ok(Duration::seconds(total_seconds))
}

pub async fn execute(args: TokenArgs) -> Result<()> {
    match args.command {
        TokenCommands::Generate(gen_args) => generate_token(gen_args).await,
        TokenCommands::List(list_args) => list_tokens(list_args).await,
        TokenCommands::Revoke(revoke_args) => revoke_token(revoke_args).await,
    }
}

async fn generate_token(args: GenerateArgs) -> Result<()> {
    output::info("Generating join token...");

    // Parse TTL
    let ttl = parse_duration(&args.ttl)?;

    // Generate token
    let mut token = JoinToken::generate(ttl, args.uses, args.capabilities.clone());

    if let Some(desc) = args.description {
        token = token.with_description(desc);
    }

    // Get cluster host
    let cluster_host = args
        .cluster_host
        .unwrap_or_else(|| "coordinator.stratoswarm.com".to_string());

    // Display token information
    output::success("Token generated successfully!");
    println!();

    output::header("Token Details");
    output::kv("Token ID", &token.id.to_string());
    output::kv("Token", &token.token);
    output::kv(
        "Expires",
        &token.expires_at.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
    );
    output::kv(
        "Max Uses",
        if token.max_uses == 0 {
            "Unlimited".to_string()
        } else {
            token.max_uses.to_string()
        }
        .as_str(),
    );

    if !token.capabilities.is_empty() {
        output::kv("Capabilities", &token.capabilities.join(", "));
    }

    println!();
    output::header("Installation Command");
    println!("curl -sSL \"{}\" | bash", token.install_url(&cluster_host));

    println!();
    output::header("Or download and review first");
    println!(
        "curl -sSL \"{}\" -o install.sh && chmod +x install.sh && ./install.sh",
        token.install_url(&cluster_host)
    );

    // Try to store token in cluster's token store
    let config = CliConfig::load().unwrap_or_default();
    let client = TokenClient::new(&config);

    println!();
    match client.store_token(&token).await {
        Ok(()) => {
            output::success("Token stored in cluster successfully.");
        }
        Err(e) => {
            output::warn(&format!(
                "Could not store token in cluster: {}. Save this token securely.",
                e
            ));
        }
    }

    Ok(())
}

async fn list_tokens(args: ListArgs) -> Result<()> {
    output::info("Fetching tokens from cluster...");

    let config = CliConfig::load().unwrap_or_default();
    let client = TokenClient::new(&config);

    match client.list_tokens(args.include_expired).await {
        Ok(tokens) => {
            if tokens.is_empty() {
                output::info("No tokens found.");
                return Ok(());
            }

            if args.format == "json" {
                println!("{}", serde_json::to_string_pretty(&tokens).unwrap_or_default());
            } else {
                output::header(&format!("Active Tokens ({})", tokens.len()));
                println!();

                for token in tokens {
                    let status = if token.is_expired() {
                        "EXPIRED"
                    } else if token.is_valid() {
                        "ACTIVE"
                    } else {
                        "EXHAUSTED"
                    };

                    output::kv("Token ID", &token.id.to_string());
                    output::kv("Status", status);
                    output::kv(
                        "Expires",
                        &token.expires_at.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                    );
                    output::kv(
                        "Uses",
                        &format!(
                            "{}/{}",
                            token.current_uses,
                            if token.max_uses == 0 {
                                "unlimited".to_string()
                            } else {
                                token.max_uses.to_string()
                            }
                        ),
                    );
                    if let Some(ref desc) = token.description {
                        output::kv("Description", desc);
                    }
                    println!();
                }
            }
        }
        Err(e) => {
            output::error(&format!("Failed to list tokens: {}", e));
            output::info("Make sure the cluster is running and you have proper authentication configured.");
            output::info("Configure with: stratoswarm config set api_endpoint <URL>");
        }
    }

    Ok(())
}

async fn revoke_token(args: RevokeArgs) -> Result<()> {
    output::info(&format!("Revoking token {}...", args.token_id));

    let config = CliConfig::load().unwrap_or_default();
    let client = TokenClient::new(&config);

    match client.revoke_token(&args.token_id).await {
        Ok(()) => {
            output::success(&format!("Token {} has been revoked.", args.token_id));
        }
        Err(e) => {
            output::error(&format!("Failed to revoke token: {}", e));
            output::info("Make sure the cluster is running and you have proper authentication configured.");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration_hours() {
        let d = parse_duration("24h").unwrap();
        assert_eq!(d.num_seconds(), 86400);
    }

    #[test]
    fn test_parse_duration_days() {
        let d = parse_duration("7d").unwrap();
        assert_eq!(d.num_seconds(), 7 * 86400);
    }

    #[test]
    fn test_parse_duration_complex() {
        let d = parse_duration("1d12h30m").unwrap();
        assert_eq!(d.num_seconds(), 86400 + 12 * 3600 + 30 * 60);
    }

    #[test]
    fn test_parse_duration_minutes() {
        let d = parse_duration("90m").unwrap();
        assert_eq!(d.num_seconds(), 90 * 60);
    }

    #[test]
    fn test_token_generation() {
        let token = JoinToken::generate(Duration::hours(24), 5, vec!["gpu_workloads".to_string()]);

        assert!(!token.token.is_empty());
        assert!(!token.is_expired());
        assert!(token.is_valid());
        assert_eq!(token.max_uses, 5);
        assert_eq!(token.current_uses, 0);
    }

    #[test]
    fn test_token_install_url() {
        let token = JoinToken::generate(Duration::hours(1), 1, vec![]);

        let url = token.install_url("cluster.example.com");
        assert!(url.starts_with("https://cluster.example.com/api/v1/install?token="));
        assert!(url.contains(&token.token));
    }

    #[test]
    fn test_token_with_description() {
        let token = JoinToken::generate(Duration::hours(1), 1, vec![])
            .with_description("Test token".to_string());

        assert_eq!(token.description, Some("Test token".to_string()));
    }

    #[test]
    fn test_unlimited_uses() {
        let token = JoinToken::generate(Duration::hours(1), 0, vec![]);

        assert!(token.is_valid());
        assert_eq!(token.max_uses, 0);
    }
}
