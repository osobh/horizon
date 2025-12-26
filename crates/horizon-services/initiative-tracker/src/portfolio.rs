pub struct PortfolioManager;

impl Default for PortfolioManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PortfolioManager {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio() {
        let _pm = PortfolioManager::new();
    }
}
