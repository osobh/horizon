use crate::error::Result;
use glob::Pattern;

/// Pattern matcher for resource patterns
pub struct PatternMatcher {
    pattern: Pattern,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub fn new(pattern: &str) -> Result<Self> {
        let pattern = Pattern::new(pattern)?;
        Ok(Self { pattern })
    }

    /// Check if the given value matches the pattern
    pub fn matches(&self, value: &str) -> bool {
        self.pattern.matches(value)
    }
}

/// Match a principal against principal definitions
pub fn match_principal(
    principal_type: &str,
    principal_value: Option<&str>,
    principal_pattern: Option<&str>,
    context_user: Option<&str>,
    context_roles: &[String],
    context_teams: &[String],
) -> bool {
    match principal_type {
        "role" => {
            if let Some(value) = principal_value {
                context_roles.iter().any(|r| r == value)
            } else if let Some(pattern) = principal_pattern {
                if let Ok(matcher) = PatternMatcher::new(pattern) {
                    context_roles.iter().any(|r| matcher.matches(r))
                } else {
                    false
                }
            } else {
                false
            }
        }
        "user" => {
            if let Some(value) = principal_value {
                context_user.map(|u| u == value).unwrap_or(false)
            } else if let Some(pattern) = principal_pattern {
                if let Ok(matcher) = PatternMatcher::new(pattern) {
                    context_user.map(|u| matcher.matches(u)).unwrap_or(false)
                } else {
                    false
                }
            } else {
                false
            }
        }
        "team" => {
            if let Some(value) = principal_value {
                context_teams.iter().any(|t| t == value)
            } else if let Some(pattern) = principal_pattern {
                if let Ok(matcher) = PatternMatcher::new(pattern) {
                    context_teams.iter().any(|t| matcher.matches(t))
                } else {
                    false
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Match a resource pattern against a resource ID
pub fn match_resource(pattern: &str, resource_id: &str) -> Result<bool> {
    let matcher = PatternMatcher::new(pattern)?;
    Ok(matcher.matches(resource_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matcher_exact_match() {
        let matcher = PatternMatcher::new("jobs/123").unwrap();
        assert!(matcher.matches("jobs/123"));
        assert!(!matcher.matches("jobs/456"));
    }

    #[test]
    fn test_pattern_matcher_wildcard() {
        let matcher = PatternMatcher::new("jobs/*").unwrap();
        assert!(matcher.matches("jobs/123"));
        assert!(matcher.matches("jobs/456"));
        assert!(!matcher.matches("users/123"));
    }

    #[test]
    fn test_pattern_matcher_double_wildcard() {
        let matcher = PatternMatcher::new("jobs/**").unwrap();
        assert!(matcher.matches("jobs/123"));
        assert!(matcher.matches("jobs/team/456"));
        assert!(!matcher.matches("users/123"));
    }

    #[test]
    fn test_pattern_matcher_question_mark() {
        let matcher = PatternMatcher::new("jobs/12?").unwrap();
        assert!(matcher.matches("jobs/123"));
        assert!(matcher.matches("jobs/124"));
        assert!(!matcher.matches("jobs/12"));
        assert!(!matcher.matches("jobs/1234"));
    }

    #[test]
    fn test_match_principal_role_exact() {
        let result = match_principal(
            "role",
            Some("admin"),
            None,
            None,
            &["admin".to_string(), "user".to_string()],
            &[],
        );
        assert!(result);
    }

    #[test]
    fn test_match_principal_role_not_found() {
        let result = match_principal(
            "role",
            Some("superadmin"),
            None,
            None,
            &["admin".to_string(), "user".to_string()],
            &[],
        );
        assert!(!result);
    }

    #[test]
    fn test_match_principal_role_pattern() {
        let result = match_principal(
            "role",
            None,
            Some("*-admin"),
            None,
            &["system-admin".to_string(), "user".to_string()],
            &[],
        );
        assert!(result);
    }

    #[test]
    fn test_match_principal_user_exact() {
        let result = match_principal(
            "user",
            Some("user@example.com"),
            None,
            Some("user@example.com"),
            &[],
            &[],
        );
        assert!(result);
    }

    #[test]
    fn test_match_principal_user_not_found() {
        let result = match_principal(
            "user",
            Some("other@example.com"),
            None,
            Some("user@example.com"),
            &[],
            &[],
        );
        assert!(!result);
    }

    #[test]
    fn test_match_principal_user_pattern() {
        let result = match_principal(
            "user",
            None,
            Some("*@example.com"),
            Some("user@example.com"),
            &[],
            &[],
        );
        assert!(result);
    }

    #[test]
    fn test_match_principal_team_exact() {
        let result = match_principal(
            "team",
            Some("ml-team"),
            None,
            None,
            &[],
            &["ml-team".to_string(), "research-team".to_string()],
        );
        assert!(result);
    }

    #[test]
    fn test_match_principal_team_not_found() {
        let result = match_principal(
            "team",
            Some("engineering-team"),
            None,
            None,
            &[],
            &["ml-team".to_string(), "research-team".to_string()],
        );
        assert!(!result);
    }

    #[test]
    fn test_match_principal_team_pattern() {
        let result = match_principal(
            "team",
            None,
            Some("*-team"),
            None,
            &[],
            &["ml-team".to_string(), "research-team".to_string()],
        );
        assert!(result);
    }

    #[test]
    fn test_match_resource_exact() {
        let result = match_resource("jobs/123", "jobs/123").unwrap();
        assert!(result);
    }

    #[test]
    fn test_match_resource_wildcard() {
        let result = match_resource("jobs/*", "jobs/123").unwrap();
        assert!(result);
    }

    #[test]
    fn test_match_resource_no_match() {
        let result = match_resource("jobs/*", "users/123").unwrap();
        assert!(!result);
    }

    #[test]
    fn test_match_resource_invalid_pattern() {
        let result = match_resource("[invalid", "jobs/123");
        assert!(result.is_err());
    }
}
