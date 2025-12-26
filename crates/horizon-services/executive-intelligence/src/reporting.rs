pub struct ReportGenerator;

impl ReportGenerator {
    pub fn generate_daily_digest() -> String {
        "Daily Digest".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daily_digest() {
        let digest = ReportGenerator::generate_daily_digest();
        assert!(!digest.is_empty());
    }
}
