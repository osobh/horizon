use crate::db::repository::MarginRepository;
use crate::error::Result;
use crate::models::*;
use rust_decimal::Decimal;
use uuid::Uuid;

#[derive(Clone)]
pub struct PricingSimulator {
    repository: MarginRepository,
}

impl PricingSimulator {
    pub fn new(repository: MarginRepository) -> Self {
        Self { repository }
    }

    pub async fn create_simulation(
        &self,
        request: CreateSimulationRequest,
    ) -> Result<PricingSimulation> {
        self.repository.create_simulation(&request).await
    }

    pub async fn get_simulation(&self, id: Uuid) -> Result<PricingSimulation> {
        self.repository.get_simulation(id).await
    }

    pub async fn run_optimization(
        &self,
        customer_id: &str,
        current_price: Decimal,
        elasticity: Decimal,
    ) -> Result<Vec<PricingSimulation>> {
        let mut simulations = Vec::new();

        // Test different price points
        let price_changes = vec![
            (Decimal::from(5), "5% increase"),
            (Decimal::from(10), "10% increase"),
            (Decimal::from(15), "15% increase"),
            (Decimal::from(-5), "5% decrease"),
        ];

        for (change_percent, name) in price_changes {
            let simulated_price = current_price * (Decimal::ONE + change_percent / Decimal::from(100));

            let request = CreateSimulationRequest {
                customer_id: customer_id.to_string(),
                scenario_name: name.to_string(),
                current_price,
                simulated_price,
                elasticity_factor: elasticity,
            };

            let sim = self.repository.create_simulation(&request).await?;
            simulations.push(sim);
        }

        Ok(simulations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatabaseConfig;
    use crate::db::pool::create_pool;
    use rust_decimal_macros::dec;

    async fn setup_test_repo() -> MarginRepository {
        let config = DatabaseConfig {
            url: std::env::var("TEST_DATABASE_URL")
                .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/margin_test".to_string()),
            max_connections: 5,
        };

        let pool = create_pool(&config).await.unwrap();
        sqlx::query("TRUNCATE pricing_simulations CASCADE")
            .execute(&pool)
            .await
            .ok();

        MarginRepository::new(pool)
    }

    #[tokio::test]
    async fn test_create_simulation() {
        let repo = setup_test_repo().await;
        let simulator = PricingSimulator::new(repo);

        let request = CreateSimulationRequest {
            customer_id: "sim-test".to_string(),
            scenario_name: "Test scenario".to_string(),
            current_price: dec!(100),
            simulated_price: dec!(110),
            elasticity_factor: dec!(-0.3),
        };

        let sim = simulator.create_simulation(request).await.unwrap();
        assert_eq!(sim.customer_id, "sim-test");
    }

    #[tokio::test]
    async fn test_run_optimization() {
        let repo = setup_test_repo().await;
        let simulator = PricingSimulator::new(repo);

        let sims = simulator
            .run_optimization("opt-test", dec!(100), dec!(-0.3))
            .await
            .unwrap();

        assert_eq!(sims.len(), 4);
    }
}
