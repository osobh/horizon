-- Restore the original (buggy) log_asset_change function

CREATE OR REPLACE FUNCTION log_asset_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO asset_history (asset_id, operation, new_status, new_metadata, changed_by)
        VALUES (NEW.id, 'created', NEW.status, NEW.metadata, NEW.created_by);
    ELSIF TG_OP = 'UPDATE' THEN
        -- Log only if status or metadata changed
        IF OLD.status != NEW.status OR OLD.metadata != NEW.metadata THEN
            INSERT INTO asset_history (
                asset_id,
                operation,
                previous_status,
                previous_metadata,
                new_status,
                new_metadata,
                metadata_delta,
                changed_by
            ) VALUES (
                NEW.id,
                CASE
                    WHEN OLD.status != NEW.status THEN 'status_changed'
                    ELSE 'metadata_changed'
                END,
                OLD.status,
                OLD.metadata,
                NEW.status,
                NEW.metadata,
                -- Compute delta (simplified - only new fields)
                NEW.metadata,
                'system'
            );
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
