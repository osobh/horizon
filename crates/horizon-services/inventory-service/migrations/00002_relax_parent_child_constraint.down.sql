-- Restore the strict parent_child_type_check constraint

ALTER TABLE assets
DROP CONSTRAINT parent_child_type_check;

ALTER TABLE assets
ADD CONSTRAINT parent_child_type_check CHECK (
    (asset_type = 'gpu' AND parent_id IS NOT NULL) OR
    (asset_type = 'cpu' AND parent_id IS NOT NULL) OR
    (asset_type = 'nic' AND parent_id IS NOT NULL) OR
    asset_type IN ('node', 'switch', 'rack')
);
