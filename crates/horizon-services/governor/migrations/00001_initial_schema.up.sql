-- Create policies table
CREATE TABLE policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    version INT NOT NULL DEFAULT 1,
    content TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT true,

    CONSTRAINT valid_content CHECK (content != ''),
    CONSTRAINT valid_name CHECK (name != ''),
    CONSTRAINT valid_created_by CHECK (created_by != ''),
    CONSTRAINT positive_version CHECK (version > 0)
);

-- Create indexes for policies table
CREATE INDEX idx_policies_name ON policies(name);
CREATE INDEX idx_policies_enabled ON policies(enabled) WHERE enabled = true;
CREATE INDEX idx_policies_created_at ON policies(created_at DESC);

-- Create policy versions table
CREATE TABLE policy_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    version INT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) NOT NULL,

    CONSTRAINT valid_version_content CHECK (content != ''),
    CONSTRAINT valid_version_created_by CHECK (created_by != ''),
    CONSTRAINT positive_version_number CHECK (version > 0),
    UNIQUE(policy_id, version)
);

-- Create indexes for policy_versions table
CREATE INDEX idx_policy_versions_policy_id ON policy_versions(policy_id, version DESC);
CREATE INDEX idx_policy_versions_created_at ON policy_versions(created_at DESC);
