-- Drop triggers
DROP TRIGGER IF EXISTS update_cost_attributions_updated_at ON cost_attributions;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop tables
DROP TABLE IF EXISTS cost_attributions;
DROP TABLE IF EXISTS gpu_pricing;
