pub struct ContractManager;

impl Default for ContractManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ContractManager {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_manager() {
        let _cm = ContractManager::new();
    }
}
