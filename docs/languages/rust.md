# Rust Development Rules

## Overview

This document defines comprehensive coding standards, length limits, linting configurations, and Test-Driven Development (TDD) practices for Rust projects. All development follows a strict TDD approach with comprehensive testing requirements.

## Rust Coding Standards

### Style Guide and Formatting

- **Style Guide**: Official Rust Style Guide + Rust API Guidelines
- **Formatter**: rustfmt (cargo fmt)
- **Line Length**: 100 characters
- **Indentation**: 4 spaces (Rust standard)
- **Naming**: Follow Rust naming conventions strictly
- **Edition**: Rust 2021 edition minimum

### Import Organization

- Group imports: Standard library, external crates, local modules
- Use `use` statements efficiently with glob imports sparingly
- Prefer explicit imports over wildcard imports
- Group related imports with blank lines
- Sort imports alphabetically within groups

```rust
// Standard library
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

// External crates
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

// Local modules
use crate::config::AppConfig;
use crate::error::UserError;
use crate::models::User;

// Local submodules
use super::validation::validate_email;
```

### Naming Conventions

| Element          | Convention      | Example              |
| ---------------- | --------------- | -------------------- |
| Crates           | snake_case      | `user_service`       |
| Modules          | snake_case      | `user_management`    |
| Files            | snake_case      | `user_service.rs`    |
| Functions        | snake_case      | `process_user_data`  |
| Variables        | snake_case      | `user_count`         |
| Constants        | SCREAMING_SNAKE | `MAX_RETRIES`        |
| Static Variables | SCREAMING_SNAKE | `GLOBAL_CONFIG`      |
| Types (structs)  | PascalCase      | `UserService`        |
| Traits           | PascalCase      | `UserRepository`     |
| Enums            | PascalCase      | `UserStatus`         |
| Enum Variants    | PascalCase      | `UserStatus::Active` |
| Type Parameters  | PascalCase      | `T`, `UserType`      |
| Lifetimes        | lowercase       | `'a`, `'static`      |
| Macros           | snake_case!     | `debug_assert!`      |

### Rust-Specific Standards

#### Error Handling

- Use `Result<T, E>` for recoverable errors
- Use `panic!` only for unrecoverable errors or programming bugs
- Prefer `anyhow` for application errors, `thiserror` for library errors
- Always provide context when propagating errors
- Use `?` operator for error propagation

```rust
use anyhow::{Context, Result};
use thiserror::Error;

// Custom error types for libraries
#[derive(Error, Debug)]
pub enum UserError {
    #[error("User not found: {id}")]
    NotFound { id: String },

    #[error("Invalid email format: {email}")]
    InvalidEmail { email: String },

    #[error("Database error")]
    Database(#[from] sqlx::Error),

    #[error("Validation failed: {0}")]
    Validation(String),
}

// Function with proper error handling
pub async fn get_user(id: &str) -> Result<User> {
    if id.is_empty() {
        return Err(UserError::Validation("User ID cannot be empty".to_string()).into());
    }

    let user = user_repository::find_by_id(id)
        .await
        .context("Failed to query user database")?
        .ok_or_else(|| UserError::NotFound { id: id.to_string() })?;

    Ok(user)
}

// Error propagation with context
pub async fn process_user_request(request: UserRequest) -> Result<UserResponse> {
    let user = get_user(&request.user_id)
        .await
        .context("Failed to retrieve user for processing")?;

    let processed = process_user_data(&user)
        .context("Failed to process user data")?;

    Ok(UserResponse::from(processed))
}
```

#### Ownership and Borrowing

- Prefer borrowing over ownership when possible
- Use `Cow<'_, T>` for functions that may or may not need to own data
- Minimize `clone()` calls - use only when necessary
- Use `Arc<T>` and `Rc<T>` for shared ownership
- Prefer `&str` over `String` for function parameters

```rust
// Good: Borrowing for read-only operations
pub fn validate_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
}

// Good: Taking ownership when needed
pub fn normalize_email(mut email: String) -> String {
    email.make_ascii_lowercase();
    email.trim().to_string()
}

// Good: Using Cow for flexible ownership
use std::borrow::Cow;

pub fn get_user_display_name(user: &User) -> Cow<'_, str> {
    if let Some(display_name) = &user.display_name {
        Cow::Borrowed(display_name)
    } else {
        Cow::Owned(format!("{} {}", user.first_name, user.last_name))
    }
}

// Good: Shared ownership with Arc
use std::sync::Arc;

#[derive(Clone)]
pub struct UserService {
    config: Arc<AppConfig>,
    repository: Arc<dyn UserRepository>,
}
```

#### Trait Design

- Keep traits focused and cohesive
- Use associated types when the relationship is fixed
- Use generic parameters when multiple types are valid
- Implement standard traits when appropriate (`Debug`, `Clone`, `PartialEq`)

```rust
// Good: Focused trait with clear responsibility
pub trait UserRepository: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn find_by_id(&self, id: &str) -> Result<Option<User>, Self::Error>;
    async fn save(&self, user: &User) -> Result<(), Self::Error>;
    async fn delete(&self, id: &str) -> Result<bool, Self::Error>;
}

// Good: Generic trait for different data types
pub trait Validator<T> {
    type Error: std::error::Error;

    fn validate(&self, data: &T) -> Result<(), Self::Error>;
}

// Good: Implementing standard traits
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub email: String,
    pub name: String,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: chrono::DateTime<chrono::Utc>,
}
```

#### Async Programming

- Use `async`/`await` for I/O operations
- Prefer `tokio` runtime for async applications
- Use `futures` crate utilities when needed
- Handle cancellation with proper cleanup

```rust
use tokio::time::{timeout, Duration};
use futures::future::try_join_all;

// Good: Async function with timeout
pub async fn get_user_with_timeout(id: &str) -> Result<User> {
    timeout(Duration::from_secs(5), get_user(id))
        .await
        .context("User lookup timed out")?
}

// Good: Concurrent operations
pub async fn get_multiple_users(ids: &[String]) -> Result<Vec<User>> {
    let futures = ids.iter().map(|id| get_user(id));
    try_join_all(futures).await
}

// Good: Graceful shutdown handling
pub async fn run_service(mut shutdown_rx: tokio::sync::mpsc::Receiver<()>) -> Result<()> {
    let mut interval = tokio::time::interval(Duration::from_secs(30));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Periodic work
                process_background_tasks().await?;
            }
            _ = shutdown_rx.recv() => {
                info!("Received shutdown signal, cleaning up...");
                cleanup_resources().await?;
                break;
            }
        }
    }

    Ok(())
}
```

### Documentation Standards

#### Rust Doc Comments

````rust
//! User management module.
//!
//! This module provides functionality for managing user accounts, including
//! creation, validation, and persistence. It integrates with external
//! authentication providers and maintains user session state.
//!
//! # Examples
//!
//! ```rust
//! use user_service::{UserService, UserRequest};
//!
//! # tokio_test::block_on(async {
//! let service = UserService::new(config).await?;
//! let user = service.create_user(UserRequest {
//!     email: "user@example.com".to_string(),
//!     name: "John Doe".to_string(),
//! }).await?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # });
//! ```

/// Represents a user in the system.
///
/// A user contains basic identification information and metadata about
/// their account status and permissions.
///
/// # Examples
///
/// ```rust
/// use user_service::User;
/// use chrono::Utc;
///
/// let user = User {
///     id: "user123".to_string(),
///     email: "john@example.com".to_string(),
///     name: "John Doe".to_string(),
///     created_at: Utc::now(),
/// };
///
/// assert_eq!(user.email, "john@example.com");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct User {
    /// Unique identifier for the user
    pub id: String,
    /// User's email address (must be unique)
    pub email: String,
    /// User's display name
    pub name: String,
    /// Timestamp when the user account was created
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Service for managing user operations.
///
/// The `UserService` provides high-level operations for user management,
/// including creation, validation, and persistence. It coordinates between
/// the repository layer and external services.
///
/// # Examples
///
/// ```rust
/// # use user_service::{UserService, AppConfig};
/// # tokio_test::block_on(async {
/// let config = AppConfig::from_env()?;
/// let service = UserService::new(config).await?;
///
/// // Create a new user
/// let user = service.create_user("john@example.com", "John Doe").await?;
/// println!("Created user: {}", user.id);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// # });
/// ```
pub struct UserService {
    repository: Arc<dyn UserRepository>,
    validator: EmailValidator,
}

impl UserService {
    /// Creates a new `UserService` instance.
    ///
    /// This method initializes the service with the provided configuration,
    /// setting up database connections and external service clients.
    ///
    /// # Arguments
    ///
    /// * `config` - Application configuration containing database and service settings
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the initialized service or an error if
    /// initialization fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Database connection cannot be established
    /// - Configuration is invalid
    /// - Required external services are unavailable
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use user_service::{UserService, AppConfig};
    /// # tokio_test::block_on(async {
    /// let config = AppConfig::from_env()?;
    /// let service = UserService::new(config).await?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # });
    /// ```
    pub async fn new(config: AppConfig) -> Result<Self> {
        let repository = create_repository(&config.database_url).await?;
        let validator = EmailValidator::new();

        Ok(Self {
            repository: Arc::new(repository),
            validator,
        })
    }

    /// Creates a new user with the provided email and name.
    ///
    /// This method validates the input data, ensures the email is unique,
    /// and persists the new user to the database.
    ///
    /// # Arguments
    ///
    /// * `email` - The user's email address (must be valid and unique)
    /// * `name` - The user's display name (must not be empty)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created `User` with generated ID and
    /// timestamp, or an error if creation fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Email format is invalid
    /// - Email is already in use
    /// - Name is empty
    /// - Database operation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use user_service::UserService;
    /// # tokio_test::block_on(async {
    /// # let service = UserService::new(Default::default()).await?;
    /// let user = service.create_user("jane@example.com", "Jane Smith").await?;
    /// assert_eq!(user.email, "jane@example.com");
    /// assert_eq!(user.name, "Jane Smith");
    /// assert!(!user.id.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # });
    /// ```
    pub async fn create_user(&self, email: &str, name: &str) -> Result<User> {
        // Implementation...
    }
}
````

**Required Documentation**:

- All public functions, methods, types, and traits
- All modules (module-level doc comments)
- Complex internal functions
- Examples for all public APIs
- Error conditions for fallible functions

## Test-Driven Development (TDD)

### Core TDD Requirements

**MANDATORY TDD WORKFLOW:**

1. **RED PHASE**: Write failing tests FIRST, then implementation
2. **GREEN PHASE**: Write minimal code to make tests pass
3. **REFACTOR PHASE**: Improve code while keeping tests green

**Testing Requirements:**

- Write unit tests for ALL functions/methods
- Write integration tests for module interactions
- Write E2E tests for complete workflows
- Maintain minimum 85% code coverage
- All tests must pass before proceeding to next task

### Testing Framework and Structure

- **Framework**: Built-in Rust testing + additional test utilities
- **Assertion Library**: Built-in `assert!` macros + `pretty_assertions`
- **Async Testing**: `tokio-test` for async code
- **Property Testing**: `proptest` for property-based testing
- **Mocking**: `mockall` for mock objects

### Test Organization

```rust
// src/lib.rs - Library with testable code
pub mod user;
pub mod repository;
pub mod validation;

// src/user.rs - Implementation with unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_user_creation() {
        // Unit test implementation
    }
}

// tests/integration_test.rs - Integration tests
use user_service::{UserService, AppConfig};

#[tokio::test]
async fn test_user_service_integration() {
    // Integration test implementation
}

// tests/e2e_test.rs - End-to-end tests
#[tokio::test]
async fn test_complete_user_workflow() {
    // E2E test implementation
}
```

### Unit Testing Patterns

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use pretty_assertions::assert_eq;
    use std::collections::HashMap;

    // Test data builders
    fn create_test_user() -> User {
        User {
            id: "test-123".to_string(),
            email: "test@example.com".to_string(),
            name: "Test User".to_string(),
            created_at: Utc::now(),
        }
    }

    fn create_test_user_with_email(email: &str) -> User {
        User {
            email: email.to_string(),
            ..create_test_user()
        }
    }

    #[test]
    fn test_user_creation_with_valid_data() {
        // Arrange
        let email = "valid@example.com";
        let name = "Valid User";

        // Act
        let user = User::new(email, name);

        // Assert
        assert_eq!(user.email, email);
        assert_eq!(user.name, name);
        assert!(!user.id.is_empty());
        assert!(user.created_at <= Utc::now());
    }

    #[test]
    fn test_user_creation_with_invalid_email() {
        // Arrange
        let invalid_email = "not-an-email";
        let name = "Test User";

        // Act
        let result = User::new(invalid_email, name);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            UserError::InvalidEmail { email } => {
                assert_eq!(email, invalid_email);
            }
            _ => panic!("Expected InvalidEmail error"),
        }
    }

    #[test]
    fn test_email_validation_valid_emails() {
        let valid_emails = vec![
            "user@example.com",
            "test.email+tag@domain.co.uk",
            "user123@test-domain.com",
        ];

        for email in valid_emails {
            assert!(validate_email(email), "Email should be valid: {}", email);
        }
    }

    #[test]
    fn test_email_validation_invalid_emails() {
        let invalid_emails = vec![
            "not-an-email",
            "@example.com",
            "user@",
            "user name@example.com",
            "",
        ];

        for email in invalid_emails {
            assert!(!validate_email(email), "Email should be invalid: {}", email);
        }
    }

    // Property-based testing
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_user_id_generation_is_unique(
            email in r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            name in r"[A-Za-z\s]{1,50}"
        ) {
            let user1 = User::new(&email, &name).unwrap();
            let user2 = User::new(&email, &name).unwrap();

            prop_assert_ne!(user1.id, user2.id);
        }
    }

    // Async unit tests
    #[tokio::test]
    async fn test_async_user_validation() {
        // Arrange
        let user = create_test_user();
        let validator = AsyncUserValidator::new();

        // Act
        let result = validator.validate(&user).await;

        // Assert
        assert!(result.is_ok());
    }

    // Parameterized tests
    #[test]
    fn test_email_normalization() {
        let test_cases = vec![
            ("USER@EXAMPLE.COM", "user@example.com"),
            ("  user@example.com  ", "user@example.com"),
            ("User.Name@Example.Com", "user.name@example.com"),
        ];

        for (input, expected) in test_cases {
            let result = normalize_email(input);
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
    }
}
```

### Integration Testing

```rust
// tests/integration_user_service.rs
use user_service::{UserService, AppConfig, User};
use sqlx::PgPool;
use testcontainers::{clients::Cli, images::postgres::Postgres, Container};
use uuid::Uuid;

struct TestContext {
    _container: Container<'static, Postgres>,
    service: UserService,
    pool: PgPool,
}

impl TestContext {
    async fn new() -> Self {
        // Setup test database
        let docker = Cli::default();
        let container = docker.run(Postgres::default());
        let connection_string = format!(
            "postgresql://postgres:postgres@127.0.0.1:{}/postgres",
            container.get_host_port_ipv4(5432)
        );

        // Run migrations
        let pool = PgPool::connect(&connection_string).await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();

        // Create service
        let config = AppConfig {
            database_url: connection_string,
            ..Default::default()
        };
        let service = UserService::new(config).await.unwrap();

        Self {
            _container: container,
            service,
            pool,
        }
    }

    async fn cleanup(&self) {
        sqlx::query("TRUNCATE TABLE users").execute(&self.pool).await.unwrap();
    }
}

#[tokio::test]
async fn test_user_crud_operations() {
    // Arrange
    let ctx = TestContext::new().await;
    let email = "integration@example.com";
    let name = "Integration User";

    // Act - Create
    let created_user = ctx.service.create_user(email, name).await.unwrap();

    // Assert - Create
    assert_eq!(created_user.email, email);
    assert_eq!(created_user.name, name);
    assert!(!created_user.id.is_empty());

    // Act - Read
    let retrieved_user = ctx.service.get_user(&created_user.id).await.unwrap();

    // Assert - Read
    assert_eq!(retrieved_user, created_user);

    // Act - Update
    let updated_name = "Updated Name";
    let updated_user = ctx.service.update_user_name(&created_user.id, updated_name).await.unwrap();

    // Assert - Update
    assert_eq!(updated_user.name, updated_name);
    assert_eq!(updated_user.email, email);
    assert_eq!(updated_user.id, created_user.id);

    // Act - Delete
    let deleted = ctx.service.delete_user(&created_user.id).await.unwrap();

    // Assert - Delete
    assert!(deleted);
    let result = ctx.service.get_user(&created_user.id).await;
    assert!(matches!(result.unwrap_err(), UserError::NotFound { .. }));

    // Cleanup
    ctx.cleanup().await;
}

#[tokio::test]
async fn test_duplicate_email_handling() {
    // Arrange
    let ctx = TestContext::new().await;
    let email = "duplicate@example.com";

    // Act - Create first user
    let user1 = ctx.service.create_user(email, "User One").await.unwrap();

    // Act - Try to create second user with same email
    let result = ctx.service.create_user(email, "User Two").await;

    // Assert
    assert!(result.is_err());
    match result.unwrap_err() {
        UserError::DuplicateEmail { email: err_email } => {
            assert_eq!(err_email, email);
        }
        _ => panic!("Expected DuplicateEmail error"),
    }

    // Cleanup
    ctx.cleanup().await;
}

#[tokio::test]
async fn test_concurrent_user_creation() {
    use tokio::task::JoinSet;

    // Arrange
    let ctx = TestContext::new().await;
    let mut tasks = JoinSet::new();

    // Act - Create multiple users concurrently
    for i in 0..10 {
        let service = ctx.service.clone();
        tasks.spawn(async move {
            service.create_user(
                &format!("user{}@example.com", i),
                &format!("User {}", i)
            ).await
        });
    }

    // Collect results
    let mut results = Vec::new();
    while let Some(result) = tasks.join_next().await {
        results.push(result.unwrap());
    }

    // Assert
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(result.is_ok());
    }

    // Cleanup
    ctx.cleanup().await;
}
```

### End-to-End Testing

```rust
// tests/e2e_user_workflow.rs
use reqwest::Client;
use serde_json::json;
use testcontainers::{clients::Cli, images::postgres::Postgres};
use tokio::process::Command;
use uuid::Uuid;

struct E2ETestContext {
    client: Client,
    base_url: String,
    _db_container: testcontainers::Container<'static, Postgres>,
    _app_process: tokio::process::Child,
}

impl E2ETestContext {
    async fn new() -> Self {
        // Start database
        let docker = Cli::default();
        let db_container = docker.run(Postgres::default());
        let db_url = format!(
            "postgresql://postgres:postgres@127.0.0.1:{}/postgres",
            db_container.get_host_port_ipv4(5432)
        );

        // Start application
        let app_process = Command::new("cargo")
            .args(&["run", "--bin", "user-service"])
            .env("DATABASE_URL", &db_url)
            .env("PORT", "8080")
            .spawn()
            .expect("Failed to start application");

        // Wait for application to start
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        Self {
            client: Client::new(),
            base_url: "http://localhost:8080".to_string(),
            _db_container: db_container,
            _app_process: app_process,
        }
    }
}

#[tokio::test]
async fn test_complete_user_management_workflow() {
    // Arrange
    let ctx = E2ETestContext::new().await;
    let user_email = "e2e@example.com";
    let user_name = "E2E Test User";

    // Act & Assert - Create User
    let create_response = ctx.client
        .post(&format!("{}/api/users", ctx.base_url))
        .json(&json!({
            "email": user_email,
            "name": user_name
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(create_response.status(), 201);
    let created_user: serde_json::Value = create_response.json().await.unwrap();
    let user_id = created_user["id"].as_str().unwrap();

    assert_eq!(created_user["email"], user_email);
    assert_eq!(created_user["name"], user_name);

    // Act & Assert - Get User
    let get_response = ctx.client
        .get(&format!("{}/api/users/{}", ctx.base_url, user_id))
        .send()
        .await
        .unwrap();

    assert_eq!(get_response.status(), 200);
    let retrieved_user: serde_json::Value = get_response.json().await.unwrap();
    assert_eq!(retrieved_user, created_user);

    // Act & Assert - Update User
    let updated_name = "Updated E2E User";
    let update_response = ctx.client
        .put(&format!("{}/api/users/{}", ctx.base_url, user_id))
        .json(&json!({
            "name": updated_name
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(update_response.status(), 200);
    let updated_user: serde_json::Value = update_response.json().await.unwrap();
    assert_eq!(updated_user["name"], updated_name);
    assert_eq!(updated_user["email"], user_email);

    // Act & Assert - List Users
    let list_response = ctx.client
        .get(&format!("{}/api/users", ctx.base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(list_response.status(), 200);
    let users_list: serde_json::Value = list_response.json().await.unwrap();
    let users = users_list["users"].as_array().unwrap();
    assert!(users.iter().any(|u| u["id"] == user_id));

    // Act & Assert - Delete User
    let delete_response = ctx.client
        .delete(&format!("{}/api/users/{}", ctx.base_url, user_id))
        .send()
        .await
        .unwrap();

    assert_eq!(delete_response.status(), 204);

    // Act & Assert - Verify Deletion
    let get_deleted_response = ctx.client
        .get(&format!("{}/api/users/{}", ctx.base_url, user_id))
        .send()
        .await
        .unwrap();

    assert_eq!(get_deleted_response.status(), 404);
}

#[tokio::test]
async fn test_error_handling_workflow() {
    // Arrange
    let ctx = E2ETestContext::new().await;

    // Act & Assert - Invalid Email
    let invalid_email_response = ctx.client
        .post(&format!("{}/api/users", ctx.base_url))
        .json(&json!({
            "email": "invalid-email",
            "name": "Test User"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(invalid_email_response.status(), 400);
    let error: serde_json::Value = invalid_email_response.json().await.unwrap();
    assert!(error["error"].as_str().unwrap().contains("Invalid email"));

    // Act & Assert - Missing Fields
    let missing_fields_response = ctx.client
        .post(&format!("{}/api/users", ctx.base_url))
        .json(&json!({}))
        .send()
        .await
        .unwrap();

    assert_eq!(missing_fields_response.status(), 400);

    // Act & Assert - Get Non-existent User
    let fake_id = Uuid::new_v4().to_string();
    let not_found_response = ctx.client
        .get(&format!("{}/api/users/{}", ctx.base_url, fake_id))
        .send()
        .await
        .unwrap();

    assert_eq!(not_found_response.status(), 404);
}
```

### Mocking with mockall

```rust
use mockall::{automock, predicate::*};

#[automock]
#[async_trait::async_trait]
pub trait UserRepository: Send + Sync {
    async fn find_by_id(&self, id: &str) -> Result<Option<User>>;
    async fn find_by_email(&self, email: &str) -> Result<Option<User>>;
    async fn save(&self, user: &User) -> Result<()>;
    async fn delete(&self, id: &str) -> Result<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::predicate;

    #[tokio::test]
    async fn test_user_service_with_mock_repository() {
        // Arrange
        let mut mock_repo = MockUserRepository::new();

        // Set up expectations
        mock_repo
            .expect_find_by_email()
            .with(predicate::eq("test@example.com"))
            .times(1)
            .returning(|_| Ok(None));

        mock_repo
            .expect_save()
            .with(predicate::function(|user: &User| {
                user.email == "test@example.com" && user.name == "Test User"
            }))
            .times(1)
            .returning(|_| Ok(()));

        let service = UserService::new(Arc::new(mock_repo));

        // Act
        let result = service.create_user("test@example.com", "Test User").await;

        // Assert
        assert!(result.is_ok());
        let user = result.unwrap();
        assert_eq!(user.email, "test@example.com");
        assert_eq!(user.name, "Test User");
    }

    #[tokio::test]
    async fn test_duplicate_email_detection() {
        // Arrange
        let mut mock_repo = MockUserRepository::new();
        let existing_user = User {
            id: "existing-123".to_string(),
            email: "existing@example.com".to_string(),
            name: "Existing User".to_string(),
            created_at: chrono::Utc::now(),
        };

        mock_repo
            .expect_find_by_email()
            .with(predicate::eq("existing@example.com"))
            .times(1)
            .returning(move |_| Ok(Some(existing_user.clone())));

        let service = UserService::new(Arc::new(mock_repo));

        // Act
        let result = service.create_user("existing@example.com", "New User").await;

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            UserError::DuplicateEmail { email } => {
                assert_eq!(email, "existing@example.com");
            }
            _ => panic!("Expected DuplicateEmail error"),
        }
    }
}
```

## Length Restrictions

### Files

- **Maximum Lines**: 750
- **Enforcement**: Strict (as per existing requirement)
- **Exceptions**:
  - Generated code files (protobuf, procedural macros)
  - Large integration test files with comprehensive scenarios
  - Configuration files with extensive documentation

**Refactoring Strategies**:

- Split large modules into submodules
- Extract related functionality into separate files
- Use `mod.rs` files to organize module hierarchies
- Move test utilities to separate `test_utils` module
- Extract constants and type definitions to separate files

### Functions

- **Maximum Lines**: 80
- **Enforcement**: Strict
- **Exceptions**:
  - Complex state machines with detailed comments
  - Large integration test functions with comprehensive scenarios
  - Generated code from macros

**Refactoring Strategies**:

- Extract helper functions
- Use composition and iterator chains
- Break into smaller, single-purpose functions
- Use closures for inline operations
- Consider using the builder pattern for complex construction

### Structs and Enums

- **Maximum Fields per Struct**: 15
- **Maximum Variants per Enum**: 20
- **Guidance**: Prefer composition over large structs, use tuple structs when appropriate

### Implementation Blocks

- **Maximum Methods per Implementation**: 20
- **Guidance**: Use trait implementations to organize related functionality

## Linting Configuration

### Primary Tools

- **Primary Linter**: Clippy (cargo clippy)
- **Code Formatter**: rustfmt (cargo fmt)
- **Additional Tools**:
  - cargo audit (security vulnerabilities)
  - cargo deny (dependency checking)
  - cargo machete (unused dependencies)

### Clippy Configuration

**Configuration** (`clippy.toml`):

```toml
# Clippy configuration
avoid-breaking-exported-api = false
msrv = "1.70.0"

# Allow certain lints that may be too restrictive
allow = [
    "clippy::module_name_repetitions",
    "clippy::must_use_candidate",
]

# Warn on additional lints beyond default
warn = [
    "clippy::cargo",
    "clippy::pedantic",
    "clippy::nursery",
]

# Deny certain problematic patterns
deny = [
    "clippy::unwrap_used",
    "clippy::expect_used",
    "clippy::panic",
    "clippy::unimplemented",
    "clippy::todo",
]
```

**Cargo.toml linting configuration**:

```toml
[lints.rust]
unsafe_code = "forbid"
missing_docs = "warn"
missing_debug_implementations = "warn"
missing_copy_implementations = "warn"
trivial_casts = "warn"
trivial_numeric_casts = "warn"
unused_import_braces = "warn"
unused_qualifications = "warn"

[lints.clippy]
# Restriction lints - things we want to forbid
dbg_macro = "deny"
todo = "deny"
unimplemented = "deny"
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
print_stdout = "deny"
print_stderr = "deny"

# Pedantic lints - high quality code
pedantic = "warn"
nursery = "warn"
cargo = "warn"

# Allow some pedantic lints that are too noisy
module_name_repetitions = "allow"
must_use_candidate = "allow"
similar_names = "allow"
too_many_lines = "allow"  # We handle this with our own length limits
```

### rustfmt Configuration

**Configuration** (`rustfmt.toml`):

```toml
# Edition and basic formatting
edition = "2021"
max_width = 100
hard_tabs = false
tab_spaces = 4

# Import organization
imports_granularity = "Crate"
imports_layout = "Mixed"
group_imports = "StdExternalCrate"
reorder_imports = true

# Function and control flow formatting
fn_args_layout = "Tall"
brace_style = "SameLineWhere"
control_brace_style = "AlwaysSameLine"
indent_style = "Block"

# String and array formatting
format_strings = true
format_macro_matchers = true
normalize_comments = true
wrap_comments = true
comment_width = 80

# Advanced formatting
use_small_heuristics = "Default"
newline_style = "Unix"
match_block_trailing_comma = true
trailing_comma = "Vertical"
trailing_semicolon = true

# Unstable features (require nightly)
# imports_granularity = "Item"
# group_imports = "StdExternalCrate"
```

### Cargo Deny Configuration

**Configuration** (`.cargo/deny.toml`):

```toml
[graph]
targets = [
    { triple = "x86_64-unknown-linux-gnu" },
    { triple = "x86_64-apple-darwin" },
    { triple = "x86_64-pc-windows-msvc" },
]

[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "warn"
notice = "warn"
ignore = []

[licenses]
unlicensed = "deny"
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unicode-DFS-2016",
]
deny = [
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-1.0",
    "AGPL-3.0",
]
copyleft = "warn"
confidence-threshold = 0.8

[bans]
multiple-versions = "warn"
wildcards = "allow"
highlight = "all"
workspace-default-features = "allow"
external-default-features = "allow"

deny = [
    # Deny specific problematic crates
    { name = "openssl", reason = "Use rustls instead" },
]

skip = [
    # Allow multiple versions of these crates
    { name = "windows-sys" },
]

skip-tree = [
    # Skip entire dependency trees
]

[sources]
unknown-registry = "warn"
unknown-git = "warn"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

### Pre-commit Hooks

**Configuration** (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy
        language: system
        types: [rust]
        pass_filenames: false
        args: [--all-targets, --all-features, --, -D, warnings]

      - id: cargo-test
        name: cargo test
        entry: cargo test
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-audit
        name: cargo audit
        entry: cargo audit
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-deny
        name: cargo deny
        entry: cargo deny
        language: system
        types: [rust]
        pass_filenames: false
        args: [check]
```

### Makefile Integration

**Makefile**:

```makefile
.PHONY: fmt check test lint audit deny clean build

# Format code
fmt:
	cargo fmt

# Check formatting without making changes
fmt-check:
	cargo fmt -- --check

# Basic check (compilation)
check:
	cargo check --all-targets --all-features

# Run all tests
test:
	cargo test --all-targets --all-features

# Run tests with coverage
test-coverage:
	cargo tarpaulin --out html --output-dir coverage

# Lint with clippy
lint:
	cargo clippy --all-targets --all-features -- -D warnings

# Security audit
audit:
	cargo audit

# Check dependencies
deny:
	cargo deny check

# Complete quality check
quality: fmt-check check lint test audit deny
	@echo "All quality checks passed!"

# Build release
build:
	cargo build --release

# Clean build artifacts
clean:
	cargo clean

# Development workflow
dev: fmt check test
	@echo "Development checks complete!"

# CI workflow
ci: fmt-check check lint test audit deny
	@echo "CI checks complete!"

# Install development tools
install-tools:
	cargo install cargo-audit
	cargo install cargo-deny
	cargo install cargo-tarpaulin
	cargo install cargo-machete
```

## Quality Gates

### Pre-commit Requirements (Enforced by TDD)

- `cargo fmt` must be run after any code changes
- `cargo clippy` must pass with zero warnings
- `cargo test` must pass with 100% success rate
- All unit tests must exist for new functions/methods
- Integration tests must cover module interactions
- E2E tests must cover complete workflows
- Code coverage minimum 85%
- Security audit passes (cargo audit)
- Dependency check passes (cargo deny)

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: Rust CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  fmt:
    name: Formatting
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - name: Check formatting
        run: cargo fmt -- --check

  clippy:
    name: Clippy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - uses: Swatinem/rust-cache@v2
      - name: Run Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

  test:
    name: Test Suite
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macOS-latest]
        rust: [stable, beta, nightly]
        exclude:
          - os: windows-latest
            rust: nightly
          - os: macOS-latest
            rust: nightly

    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
      - uses: Swatinem/rust-cache@v2
      - name: Run tests
        run: cargo test --all-targets --all-features

  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin
      - name: Generate coverage
        run: cargo tarpaulin --out xml --output-dir coverage
      - name: Upload to codecov.io
        uses: codecov/codecov-action@v3
        with:
          file: coverage/cobertura.xml

  audit:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Install cargo-audit
        run: cargo install cargo-audit
      - name: Run audit
        run: cargo audit

  deny:
    name: Dependency Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Install cargo-deny
        run: cargo install cargo-deny
      - name: Run cargo-deny
        run: cargo deny check

  integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Run integration tests
        run: cargo test --test integration_*
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db

  e2e:
    name: End-to-End Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build application
        run: cargo build --release
      - name: Run E2E tests
        run: cargo test --test e2e_*
```

## Project Structure

### Recommended Directory Layout

```
project/
├── src/                    # Source code
│   ├── lib.rs             # Library root
│   ├── main.rs            # Binary entry point (if applicable)
│   ├── user/              # User domain module
│   │   ├── mod.rs
│   │   ├── model.rs
│   │   ├── service.rs
│   │   ├── repository.rs
│   │   └── validation.rs
│   ├── config/            # Configuration
│   │   ├── mod.rs
│   │   └── settings.rs
│   ├── error/             # Error types
│   │   ├── mod.rs
│   │   └── user_error.rs
│   └── utils/             # Utility functions
│       ├── mod.rs
│       └── helpers.rs
├── tests/                 # Integration tests
│   ├── integration_user_service.rs
│   ├── e2e_user_workflow.rs
│   └── common/           # Test utilities
│       ├── mod.rs
│       └── test_helpers.rs
├── examples/              # Example code
│   └── basic_usage.rs
├── benches/              # Benchmarks
│   └── user_benchmarks.rs
├── docs/                 # Documentation
│   └── api.md
├── migrations/           # Database migrations (if applicable)
│   └── 001_create_users.sql
├── .cargo/               # Cargo configuration
│   └── deny.toml
├── .github/              # GitHub workflows
│   └── workflows/
│       └── ci.yml
├── Cargo.toml           # Project manifest
├── Cargo.lock           # Dependency lock file
├── clippy.toml          # Clippy configuration
├── rustfmt.toml         # rustfmt configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── Makefile             # Build automation
└── README.md            # Project documentation
```

## Error Handling Patterns

### Library Error Design

```rust
// src/error/mod.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum UserError {
    #[error("User not found: {id}")]
    NotFound { id: String },

    #[error("User with email '{email}' already exists")]
    DuplicateEmail { email: String },

    #[error("Invalid email format: {email}")]
    InvalidEmail { email: String },

    #[error("Validation failed: {message}")]
    Validation { message: String },

    #[error("Database error: {source}")]
    Database {
        #[from]
        source: sqlx::Error,
    },

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Network error: {source}")]
    Network {
        #[from]
        source: reqwest::Error,
    },
}

impl UserError {
    pub fn validation<T: Into<String>>(message: T) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    pub fn config<T: Into<String>>(message: T) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            UserError::InvalidEmail { .. }
                | UserError::Validation { .. }
                | UserError::NotFound { .. }
        )
    }

    pub fn is_server_error(&self) -> bool {
        !self.is_client_error()
    }
}

// Result type alias for convenience
pub type UserResult<T> = Result<T, UserError>;
```

### Application Error Design

```rust
// For applications, use anyhow for error handling
use anyhow::{Context, Result};

pub async fn run_application() -> Result<()> {
    let config = load_config()
        .context("Failed to load application configuration")?;

    let database = connect_database(&config.database_url)
        .await
        .context("Failed to connect to database")?;

    let service = UserService::new(database)
        .context("Failed to initialize user service")?;

    start_server(service)
        .await
        .context("Failed to start HTTP server")?;

    Ok(())
}
```

## Performance Considerations

### Optimization Guidelines

- Use `&str` instead of `String` for function parameters when possible
- Prefer iterators over collecting into vectors when chaining operations
- Use `Cow<'_, str>` for functions that may or may not need to own data
- Consider using `Arc<T>` for shared data instead of cloning
- Use `tokio::spawn` for CPU-intensive tasks in async contexts

### Memory Management

```rust
// Good: Efficient string handling
pub fn process_text(input: &str) -> Cow<'_, str> {
    if input.trim() == input {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.trim().to_string())
    }
}

// Good: Shared ownership
#[derive(Clone)]
pub struct UserService {
    config: Arc<AppConfig>,
    cache: Arc<RwLock<HashMap<String, User>>>,
}

// Good: Iterator chains instead of intermediate collections
pub fn get_active_user_emails(users: &[User]) -> Vec<String> {
    users
        .iter()
        .filter(|user| user.is_active)
        .map(|user| user.email.clone())
        .collect()
}
```

### Benchmarking

```rust
// benches/user_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use user_service::{User, UserService};

fn benchmark_user_creation(c: &mut Criterion) {
    c.bench_function("user creation", |b| {
        b.iter(|| {
            User::new(
                black_box("test@example.com"),
                black_box("Test User")
            )
        })
    });
}

fn benchmark_email_validation(c: &mut Criterion) {
    let emails = vec![
        "valid@example.com",
        "another@test.org",
        "user123@domain.co.uk",
    ];

    c.bench_function("email validation", |b| {
        b.iter(|| {
            for email in &emails {
                black_box(validate_email(black_box(email)));
            }
        })
    });
}

criterion_group!(benches, benchmark_user_creation, benchmark_email_validation);
criterion_main!(benches);
```

## Security Considerations

### Input Validation

```rust
use regex::Regex;
use std::sync::OnceLock;

fn email_regex() -> &'static Regex {
    static EMAIL_REGEX: OnceLock<Regex> = OnceLock::new();
    EMAIL_REGEX.get_or_init(|| {
        Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
            .expect("Invalid email regex")
    })
}

pub fn validate_email(email: &str) -> bool {
    email_regex().is_match(email)
}

pub fn sanitize_user_input(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?".contains(*c))
        .collect()
}
```

### Secrets Management

```rust
use secrecy::{Secret, ExposeSecret};
use zeroize::Zeroize;

#[derive(Zeroize)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: Secret<String>,
    pub database: String,
}

impl DatabaseConfig {
    pub fn connection_string(&self) -> Secret<String> {
        let conn_str = format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.username,
            self.password.expose_secret(),
            self.host,
            self.port,
            self.database
        );
        Secret::new(conn_str)
    }
}
```

---

_This document serves as the comprehensive coding standard for Rust projects with mandatory Test-Driven Development. All code must follow the TDD workflow: write tests FIRST, then implementation, maintaining the quality gates and standards defined herein._
