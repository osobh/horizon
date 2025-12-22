pub struct InitiativeRegistry;

impl Default for InitiativeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl InitiativeRegistry {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry() {
        let _reg = InitiativeRegistry::new();
    }
}
