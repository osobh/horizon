pub struct Recommender;

impl Recommender {
    pub fn generate_recommendation() -> String {
        "recommendation".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_recommendation() {
        let rec = Recommender::generate_recommendation();
        assert!(!rec.is_empty());
    }
}
