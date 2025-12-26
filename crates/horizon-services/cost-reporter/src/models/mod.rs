pub mod finance;
pub mod report;
pub mod summary;
pub mod trend;

// Export non-conflicting types explicitly
// Note: CostBreakdown, ChargebackReport, and ChargebackLineItem have multiple definitions
// Use module-qualified paths to disambiguate (e.g. finance::CostBreakdown vs summary::CostBreakdown)
