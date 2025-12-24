# HPC-AI Ephemeral Access System - Next Steps

## Completed Work (Phases 1-3)

### Phase 1: Foundation
- [x] Created `hpc-ephemeral-identity` crate with 144 tests
  - `identity.rs` - EphemeralIdentity with state machine
  - `token.rs` - EphemeralToken with Ed25519 signing
  - `capabilities.rs` - CapabilitySet with rate limits and time restrictions
  - `invitation.rs` - InvitationLink with two-factor redemption
  - `service.rs` - EphemeralIdentityService with full CRUD
  - `error.rs` - Comprehensive error types
- [x] Added ephemeral quota types to Quota Manager
  - `models/ephemeral.rs` - EphemeralQuota with burst support
  - `models/time_window.rs` - TimeWindow with timezone support
  - `models/resource_pool.rs` - ResourcePool with approval workflows
  - `models/cost_tracking.rs` - Cost attribution types
- [x] Added EphemeralOrchestrator to hpc-channels (97 tests)

### Phase 2: Identity & Tokens
- [x] Integrated with Zero-Trust SessionManager
  - `integration/session_adapter.rs` - EphemeralSessionAdapter
  - Risk-based access control (Low/Medium/High/Critical)
  - Device binding validation
  - Session timeout calculation based on identity expiry

### Phase 3: Resource Allocation
- [x] Implemented EphemeralQuotaService with burst logic
  - `service/ephemeral_quota_service.rs`
  - Time-bounded quota management
  - Burst capacity with configurable multipliers
  - Reserve/commit/release operations
  - Automatic expiry handling
- [x] Implemented ResourcePoolService for bounties
  - `service/pool_service.rs`
  - Pool lifecycle management
  - Allocation approval workflows
  - Auto-approval by email domain
  - Usage tracking and statistics
- [x] Created Governor policy templates
  - `policies/ephemeral-quota-access.yaml`
  - `policies/ephemeral-identity-access.yaml`
  - `policies/bounty-pool-access.yaml`
  - `policies/ephemeral-federated-access.yaml`
  - `ephemeral/policy_generator.rs` - Dynamic policy generation

---

## Remaining Work

### Phase 4: P2P & Collaboration

#### 4.1 Nebula - EphemeralPeerRegistry
**Location**: `/10-nebula/crates/nebula-core/src/ephemeral.rs`

```rust
// Types to implement
struct EphemeralPeerRegistry {
    peers: HashMap<String, EphemeralPeer>,
    sessions: HashMap<String, EphemeralSession>,
}

struct EphemeralPeer {
    peer_id: String,
    ephemeral_identity_id: Uuid,
    display_name: Option<String>,
    connection_mode: ConnectionMode,  // P2P | Relay
    quality: ConnectionQuality,
    allowed_channels: Vec<String>,
    joined_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
}

// Key functionality
impl EphemeralPeerRegistry {
    fn register_peer(&mut self, identity: &EphemeralIdentity) -> Result<EphemeralPeer>;
    fn validate_channel_access(&self, peer_id: &str, channel: &str) -> bool;
    fn get_session_peers(&self, session_id: &str) -> Vec<&EphemeralPeer>;
    fn cleanup_expired(&mut self) -> Vec<String>;
}
```

**Tasks**:
1. Create `ephemeral.rs` module in nebula-core
2. Implement EphemeralPeerRegistry with peer lifecycle
3. Add channel-level access control (ScopedDataChannel)
4. Integrate with existing Nebula peer management
5. Add tests for peer registration, validation, cleanup

#### 4.2 Nebula - ScopedDataChannel
**Location**: `/10-nebula/crates/nebula-core/src/scoped_channel.rs`

```rust
struct ScopedDataChannel {
    channel_id: String,
    allowed_data_types: HashSet<String>,  // "gradients", "cursors", "edits"
    max_message_size: u64,
    rate_limit: RateLimiter,
    ephemeral_only: bool,
}

impl ScopedDataChannel {
    fn can_send(&self, peer: &EphemeralPeer, data_type: &str, size: u64) -> bool;
    fn record_message(&mut self, peer_id: &str, size: u64);
}
```

**Tasks**:
1. Create ScopedDataChannel with data type filtering
2. Implement rate limiting per ephemeral peer
3. Add message size validation
4. Integrate with Nebula's channel system

#### 4.3 Warp - EphemeralRelayAuth
**Location**: `/05-warp/crates/portal-hub/src/ephemeral_auth.rs`

```rust
struct EphemeralRelayAuth {
    validator: TokenValidator,
    allowed_peers: HashMap<String, EphemeralRelayPermission>,
}

struct EphemeralRelayPermission {
    ephemeral_identity_id: Uuid,
    allowed_targets: Vec<String>,  // Peer IDs they can relay to
    priority: RelayPriority,       // Low | Normal | High
    bandwidth_limit: Option<u64>,
    expires_at: DateTime<Utc>,
}

impl EphemeralRelayAuth {
    fn authorize_relay(&self, token: &str, target: &str) -> Result<RelayAuthorization>;
    fn validate_ongoing(&self, peer_id: &str) -> bool;
    fn cleanup_expired(&mut self);
}
```

**Tasks**:
1. Create ephemeral_auth.rs in portal-hub
2. Implement token validation for relay requests
3. Add target-based authorization (who can relay to whom)
4. Implement bandwidth limiting for ephemeral peers
5. Add priority-based relay queuing
6. Integrate with Warp's existing relay system

#### 4.4 Horizon - EphemeralBridge (Tauri)
**Location**: `/11-horizon/src-tauri/src/ephemeral_bridge.rs`

```rust
pub struct EphemeralBridge {
    identity_service: EphemeralIdentityService,
    session_adapter: EphemeralSessionAdapter,
    active_sessions: HashMap<String, EphemeralSession>,
}

// Tauri commands to implement
#[tauri::command]
async fn create_ephemeral_session(
    state: State<'_, EphemeralBridge>,
    session_type: String,
    permissions: EphemeralPermissions,
    ttl_minutes: i64,
) -> Result<CreateSessionResponse, String>;

#[tauri::command]
async fn get_invite_url(
    state: State<'_, EphemeralBridge>,
    session_id: String,
) -> Result<InviteUrlResponse, String>;

#[tauri::command]
async fn join_ephemeral_session(
    state: State<'_, EphemeralBridge>,
    invite_token: String,
    redemption_code: String,
    device_info: DeviceInfo,
) -> Result<JoinSessionResponse, String>;

#[tauri::command]
async fn get_ephemeral_presence(
    state: State<'_, EphemeralBridge>,
    session_id: String,
) -> Result<Vec<EphemeralParticipant>, String>;

#[tauri::command]
async fn end_ephemeral_session(
    state: State<'_, EphemeralBridge>,
    session_id: String,
) -> Result<(), String>;
```

**Tasks**:
1. Create ephemeral_bridge.rs with EphemeralBridge struct
2. Implement all Tauri commands
3. Add commands/ephemeral.rs for command module
4. Update src-tauri/src/state.rs to include EphemeralBridge
5. Add event emitters for session lifecycle events
6. Wire up to frontend via Tauri invoke

---

### Phase 5: Frontend & Polish

#### 5.1 Horizon Frontend - Ephemeral Store
**Location**: `/11-horizon/web/src/stores/ephemeralStore.ts`

```typescript
interface EphemeralStore {
  // State
  activeSessions: Map<string, EphemeralSession>;
  pendingInvites: Map<string, PendingInvite>;
  ephemeralParticipants: Map<string, EphemeralParticipant[]>;

  // Actions
  createSession(type: SessionType, permissions: Permissions): Promise<Session>;
  generateInviteLink(sessionId: string): Promise<InviteLink>;
  joinSession(token: string, code: string): Promise<void>;
  leaveSession(sessionId: string): Promise<void>;

  // Subscriptions
  subscribeToPresence(sessionId: string): void;
  subscribeToQuality(sessionId: string): void;
}
```

**Tasks**:
1. Create ephemeralStore.ts with Zustand/Pinia
2. Implement Tauri invoke wrappers
3. Add real-time presence updates via Tauri events
4. Handle connection quality monitoring

#### 5.2 Horizon Frontend - Components
**Location**: `/11-horizon/web/src/components/ephemeral/`

| Component | Purpose | Priority |
|-----------|---------|----------|
| `InviteLinkGenerator.tsx` | Create invite URLs with options | High |
| `InviteLinkShare.tsx` | Share dialog with QR code, copy button | High |
| `JoinPage.tsx` | Landing page for invite redemption | High |
| `RedemptionForm.tsx` | Enter redemption code + device info | High |
| `EphemeralPresenceBar.tsx` | Shows ephemeral collaborators in session | Medium |
| `ParticipantBadge.tsx` | Individual participant with status | Medium |
| `ConnectionQualityIndicator.tsx` | P2P/Relay status, latency | Medium |
| `SessionExpiryWarning.tsx` | Countdown timer, extend option | Medium |
| `EphemeralSettingsPanel.tsx` | Manage active ephemeral sessions | Low |

**Tasks**:
1. Create InviteLinkGenerator with session type selection
2. Create InviteLinkShare with QR code generation (use qrcode library)
3. Create JoinPage as a standalone route (/join/:token)
4. Create RedemptionForm with code input and device fingerprinting
5. Create EphemeralPresenceBar for collaboration overlay
6. Add connection quality indicators
7. Implement session expiry warnings with extension option

#### 5.3 Background Workers

**Expiry Worker** (`/06-stratoswarm/crates/horizon-services/quota-manager/src/workers/`):
```rust
struct ExpiryWorker {
    ephemeral_quota_service: EphemeralQuotaService,
    pool_service: ResourcePoolService,
    interval: Duration,
}

impl ExpiryWorker {
    async fn run(&self) {
        loop {
            // Process quota expirations
            self.ephemeral_quota_service.process_expirations().await;

            // Process pool expirations
            self.pool_service.process_pool_expirations().await;
            self.pool_service.process_allocation_expirations().await;

            // Cleanup old expired entries
            self.ephemeral_quota_service.cleanup_expired(24).await;

            tokio::time::sleep(self.interval).await;
        }
    }
}
```

**Identity Cleanup Worker** (`/06-stratoswarm/crates/hpc-foundation/hpc-ephemeral-identity/src/workers/`):
```rust
struct IdentityCleanupWorker {
    identity_service: EphemeralIdentityService,
    interval: Duration,
}

impl IdentityCleanupWorker {
    async fn run(&self) {
        loop {
            // Expire pending invitations
            self.identity_service.cleanup().await;

            // Purge old expired identities
            self.identity_service.purge_expired().await;

            tokio::time::sleep(self.interval).await;
        }
    }
}
```

**Tasks**:
1. Create workers module in quota-manager
2. Implement ExpiryWorker with configurable interval
3. Create workers module in hpc-ephemeral-identity
4. Implement IdentityCleanupWorker
5. Add worker startup to service main.rs files

---

### Database Migrations

#### Quota Manager Migrations
**Location**: `/06-stratoswarm/crates/horizon-services/quota-manager/migrations/`

```sql
-- XXXXX_ephemeral_quotas.up.sql
CREATE TABLE ephemeral_quotas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quota_id UUID REFERENCES quotas(id),
    ephemeral_identity_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    sponsor_id VARCHAR(255) NOT NULL,
    beneficiary_id VARCHAR(255) NOT NULL,
    resource_type TEXT NOT NULL,
    limit_value DECIMAL(20,6) NOT NULL,
    used_value DECIMAL(20,6) NOT NULL DEFAULT 0,
    reserved_value DECIMAL(20,6) NOT NULL DEFAULT 0,
    starts_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    time_window_id UUID,
    burst_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    burst_multiplier DECIMAL(10,4) NOT NULL DEFAULT 1.0,
    pool_id UUID,
    actual_cost DECIMAL(20,6) NOT NULL DEFAULT 0,
    cost_rate DECIMAL(20,6) NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    status_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ephemeral_quotas_identity ON ephemeral_quotas(ephemeral_identity_id);
CREATE INDEX idx_ephemeral_quotas_sponsor ON ephemeral_quotas(sponsor_id);
CREATE INDEX idx_ephemeral_quotas_beneficiary ON ephemeral_quotas(beneficiary_id);
CREATE INDEX idx_ephemeral_quotas_expires ON ephemeral_quotas(expires_at);
CREATE INDEX idx_ephemeral_quotas_status ON ephemeral_quotas(status);

-- XXXXX_resource_pools.up.sql
CREATE TABLE resource_pools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tenant_id UUID NOT NULL,
    pool_type TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    total_limit DECIMAL(20,6) NOT NULL,
    allocated DECIMAL(20,6) NOT NULL DEFAULT 0,
    reserved DECIMAL(20,6) NOT NULL DEFAULT 0,
    max_allocation_per_user DECIMAL(20,6) NOT NULL,
    min_allocation_per_request DECIMAL(20,6) NOT NULL DEFAULT 1,
    requires_approval BOOLEAN NOT NULL DEFAULT FALSE,
    auto_approve_domains TEXT[],
    max_concurrent_users INTEGER,
    current_users INTEGER NOT NULL DEFAULT 0,
    starts_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active',
    sponsor_id UUID,
    time_window_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE pool_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pool_id UUID NOT NULL REFERENCES resource_pools(id),
    user_id VARCHAR(255) NOT NULL,
    ephemeral_identity_id UUID,
    allocated_amount DECIMAL(20,6) NOT NULL,
    used_amount DECIMAL(20,6) NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    purpose TEXT,
    approved_by UUID,
    approved_at TIMESTAMPTZ,
    released_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- XXXXX_time_windows.up.sql
CREATE TABLE time_windows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    timezone VARCHAR(50) NOT NULL DEFAULT 'UTC',
    days_of_week INTEGER[] NOT NULL DEFAULT '{0,1,2,3,4,5,6}',
    blackout_dates DATE[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Tasks**:
1. Create migration files with proper numbering
2. Add down migrations for rollback
3. Test migrations against local PostgreSQL
4. Update repository methods to use database

---

### Integration Tasks

#### hpc-channels Updates
**Location**: `/01-hpc-channels/src/`

1. **messages.rs** - Add EphemeralMessage enum variants:
```rust
pub enum EphemeralMessage {
    InvitationCreated { invitation_id: Uuid, sponsor_id: Uuid, expires_at: DateTime<Utc> },
    InvitationRedeemed { invitation_id: Uuid, identity_id: Uuid },
    IdentityActivated { identity_id: Uuid, capabilities: Vec<String> },
    IdentityRevoked { identity_id: Uuid, reason: String },
    EphemeralQuotaCreated { quota_id: Uuid, beneficiary_id: String },
    EphemeralQuotaExpired { quota_id: Uuid },
    TimeWindowEvaluated { quota_id: Uuid, allowed: bool },
    BurstUsed { quota_id: Uuid, amount: Decimal },
    CostRecorded { quota_id: Uuid, sponsor_id: String, cost: Decimal },
    SessionCreated { session_id: String, session_type: String },
    ParticipantJoined { session_id: String, peer_id: String },
    ParticipantLeft { session_id: String, peer_id: String },
    SessionExpired { session_id: String },
    QualityChanged { session_id: String, peer_id: String, quality: String },
}
```

2. **integrations/collaboration.rs** - Add ephemeral peer handling:
```rust
impl CollaborationOrchestrator {
    async fn handle_ephemeral_join(&self, msg: EphemeralMessage);
    async fn handle_ephemeral_leave(&self, msg: EphemeralMessage);
    async fn validate_ephemeral_action(&self, peer_id: &str, action: &str) -> bool;
}
```

#### Nebula Updates
**Location**: `/10-nebula/crates/nebula-core/src/peer.rs`

1. Extend PeerInfo for ephemeral:
```rust
pub struct PeerInfo {
    // ... existing fields ...
    pub ephemeral_identity_id: Option<Uuid>,
    pub is_ephemeral: bool,
    pub ephemeral_expires_at: Option<DateTime<Utc>>,
    pub allowed_channels: Option<Vec<String>>,
}
```

---

### Testing Requirements

#### Unit Tests
- [ ] EphemeralPeerRegistry tests (Nebula)
- [ ] ScopedDataChannel tests (Nebula)
- [ ] EphemeralRelayAuth tests (Warp)
- [ ] EphemeralBridge tests (Horizon Tauri)
- [ ] Worker tests (expiry, cleanup)

#### Integration Tests
- [ ] End-to-end invitation flow (create → share → redeem → active)
- [ ] Quota lifecycle (allocate → use → burst → expire)
- [ ] Pool allocation with approval workflow
- [ ] P2P session with ephemeral participants
- [ ] Relay fallback for ephemeral peers

#### Performance Tests
- [ ] Token validation < 5ms
- [ ] Session creation < 100ms
- [ ] Expiry cleanup processes all within 1 minute
- [ ] 1000 concurrent ephemeral sessions

---

## File Summary

### Files to Create

| Project | File | Description |
|---------|------|-------------|
| Nebula | `nebula-core/src/ephemeral.rs` | EphemeralPeerRegistry |
| Nebula | `nebula-core/src/scoped_channel.rs` | ScopedDataChannel |
| Warp | `portal-hub/src/ephemeral_auth.rs` | EphemeralRelayAuth |
| Horizon | `src-tauri/src/ephemeral_bridge.rs` | Tauri bridge |
| Horizon | `src-tauri/src/commands/ephemeral.rs` | Tauri commands |
| Horizon | `web/src/stores/ephemeralStore.ts` | Frontend store |
| Horizon | `web/src/components/ephemeral/*.tsx` | UI components (8 files) |
| Quota Manager | `src/workers/mod.rs` | Worker module |
| Quota Manager | `src/workers/expiry_worker.rs` | Expiry worker |
| Quota Manager | `migrations/*.sql` | Database migrations (3 files) |
| hpc-ephemeral-identity | `src/workers/mod.rs` | Worker module |
| hpc-ephemeral-identity | `src/workers/cleanup_worker.rs` | Cleanup worker |

### Files to Modify

| Project | File | Changes |
|---------|------|---------|
| hpc-channels | `src/messages.rs` | Add EphemeralMessage enum |
| hpc-channels | `src/integrations/collaboration.rs` | Add ephemeral handling |
| Nebula | `nebula-core/src/peer.rs` | Extend PeerInfo |
| Nebula | `nebula-core/src/lib.rs` | Export ephemeral module |
| Warp | `portal-hub/src/lib.rs` | Export ephemeral_auth |
| Horizon | `src-tauri/src/state.rs` | Add EphemeralBridge |
| Horizon | `src-tauri/src/main.rs` | Register commands |
| Quota Manager | `src/main.rs` | Start workers |

---

## Priority Order

### High Priority (Do First)
1. Database migrations (blocking for persistence)
2. EphemeralPeerRegistry in Nebula (core P2P functionality)
3. EphemeralRelayAuth in Warp (relay authorization)
4. EphemeralBridge in Horizon (Tauri integration)
5. Core frontend components (InviteLinkGenerator, JoinPage)

### Medium Priority
1. ScopedDataChannel in Nebula
2. hpc-channels message updates
3. Background workers
4. Remaining frontend components

### Low Priority (Polish)
1. Connection quality indicators
2. Session expiry warnings
3. Settings panel
4. Performance optimization

---

## Estimated Effort

| Phase | Components | Complexity |
|-------|------------|------------|
| Phase 4.1 | Nebula EphemeralPeerRegistry | Medium |
| Phase 4.2 | Nebula ScopedDataChannel | Medium |
| Phase 4.3 | Warp EphemeralRelayAuth | Medium |
| Phase 4.4 | Horizon EphemeralBridge | Medium-High |
| Phase 5.1 | Frontend Store | Medium |
| Phase 5.2 | Frontend Components | High (8 components) |
| Phase 5.3 | Background Workers | Low-Medium |
| Migrations | Database setup | Low |
| Integration | Cross-project wiring | Medium |
| Testing | Full test coverage | Medium-High |

---

## Notes

- All services currently use in-memory caches; database persistence requires migrations first
- Nebula and Warp codebases need exploration before implementation
- Frontend components should follow Horizon's existing design system
- Workers should be configurable via environment variables
- Consider feature flags for gradual rollout
