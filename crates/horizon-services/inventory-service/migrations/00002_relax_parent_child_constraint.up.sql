-- Relax the parent_child_type_check constraint to allow optional parents for GPUs, CPUs, and NICs
-- This allows more flexible asset creation while still maintaining the logical relationships

ALTER TABLE assets
DROP CONSTRAINT parent_child_type_check;

-- New constraint: parent_id is optional for all asset types
-- This allows standalone GPU/CPU/NIC creation for testing or edge cases
-- The application logic can enforce stricter rules as needed
ALTER TABLE assets
ADD CONSTRAINT parent_child_type_check CHECK (
    (parent_id IS NULL) OR
    (asset_type IN ('gpu', 'cpu', 'nic') AND parent_id IS NOT NULL) OR
    (asset_type IN ('node', 'switch', 'rack'))
);
