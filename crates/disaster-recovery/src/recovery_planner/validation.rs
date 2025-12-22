//! Recovery plan validation and testing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Validation test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTest {
    /// Test ID
    pub id: Uuid,
    /// Test name
    pub name: String,
    /// Test description
    pub description: String,
    /// Test type
    pub test_type: TestType,
    /// Expected result
    pub expected_result: String,
    /// Test timeout in seconds
    pub timeout_seconds: u64,
    /// Test parameters
    pub parameters: HashMap<String, String>,
    /// Critical test flag
    pub critical: bool,
}

/// Types of validation tests
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TestType {
    /// Connectivity test
    Connectivity,
    /// Health check
    HealthCheck,
    /// Performance benchmark
    Performance,
    /// Data integrity check
    DataIntegrity,
    /// Security validation
    Security,
    /// Compliance check
    Compliance,
    /// Custom script test
    Custom { script_path: String },
}

/// Test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Test ID
    pub test_id: Uuid,
    /// Test success status
    pub success: bool,
    /// Test output
    pub output: String,
    /// Error message if failed
    pub error: Option<String>,
    /// Test duration
    pub duration_ms: u64,
    /// Test metrics
    pub metrics: HashMap<String, f64>,
}

/// Validation test suite
pub struct ValidationSuite {
    /// Suite name
    pub name: String,
    /// Tests in the suite
    pub tests: Vec<ValidationTest>,
    /// Parallel execution flag
    pub parallel: bool,
    /// Continue on failure flag
    pub continue_on_failure: bool,
}

impl ValidationSuite {
    /// Create new validation suite
    pub fn new(name: String) -> Self {
        Self {
            name,
            tests: Vec::new(),
            parallel: false,
            continue_on_failure: false,
        }
    }

    /// Add test to suite
    pub fn add_test(&mut self, test: ValidationTest) {
        self.tests.push(test);
    }

    /// Get test by ID
    pub fn get_test(&self, test_id: &str) -> Option<&ValidationTest> {
        self.tests.iter().find(|t| t.id.to_string() == test_id)
    }

    /// Get critical tests
    pub fn critical_tests(&self) -> Vec<&ValidationTest> {
        self.tests.iter().filter(|t| t.critical).collect()
    }

    /// Validate test suite configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.tests.is_empty() {
            return Err("Test suite cannot be empty".to_string());
        }

        // Check for duplicate test names
        let mut names = std::collections::HashSet::new();
        for test in &self.tests {
            if !names.insert(&test.name) {
                return Err(format!("Duplicate test name: {}", test.name));
            }
        }

        // Validate individual tests
        for test in &self.tests {
            if test.timeout_seconds == 0 {
                return Err(format!("Test '{}' has invalid timeout", test.name));
            }
        }

        Ok(())
    }
}
