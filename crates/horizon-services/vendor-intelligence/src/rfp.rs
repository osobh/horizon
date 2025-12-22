pub struct RFPGenerator;

impl RFPGenerator {
    pub fn generate() -> String {
        "RFP Template".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfp_generation() {
        let rfp = RFPGenerator::generate();
        assert!(!rfp.is_empty());
    }
}
