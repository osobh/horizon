use crate::db::BillingRepository;
use crate::normalize::NormalizedBillingSchema;

pub struct AppState {
    pub repository: BillingRepository,
    pub schema: NormalizedBillingSchema,
}
