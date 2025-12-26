# Visual Editor

Web-based topology editor for StratoSwarm with drag-and-drop interface and real-time collaboration.

## Overview

The `visual-editor` crate provides a modern web interface for designing and managing StratoSwarm deployments visually. It features an intuitive drag-and-drop interface, real-time synchronization, and seamless integration with the StratoSwarm ecosystem. Users can visually design complex distributed systems without writing configuration files.

## Features

- **Drag-and-Drop Interface**: Visual topology design with intuitive controls
- **Real-time Collaboration**: Multiple users can edit simultaneously
- **Live Preview**: See changes reflected in the cluster immediately
- **Component Library**: Pre-built components for common patterns
- **Import/Export**: Support for .swarm DSL and other formats
- **WebSocket Updates**: Real-time status and metrics
- **GraphQL API**: Flexible querying and mutations
- **Responsive Design**: Works on desktop and tablet devices
- **Dark Mode**: Easy on the eyes during long sessions

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
├─────────────────────────────────────────────────────┤
│           GraphQL API / WebSocket Server             │
├─────────────────────────────────────────────────────┤
│              Visual Editor Backend                   │
├─────────────────────────────────────────────────────┤
│                StratoSwarm Core                      │
└─────────────────────────────────────────────────────┘
```

## Backend Usage

### Starting the Server

```rust
use visual_editor::{Server, ServerConfig};

// Configure server
let config = ServerConfig::default()
    .port(8080)
    .enable_cors()
    .enable_auth();

// Start server
let server = Server::new(config);
server.run().await?;
```

### GraphQL Schema

```graphql
type Agent {
  id: ID!
  name: String!
  type: AgentType!
  replicas: Int!
  resources: Resources!
  connections: [Connection!]!
  status: AgentStatus!
  metrics: Metrics
}

type Query {
  agents: [Agent!]!
  agent(id: ID!): Agent
  topology: Topology!
  deployments: [Deployment!]!
}

type Mutation {
  createAgent(input: CreateAgentInput!): Agent!
  updateAgent(id: ID!, input: UpdateAgentInput!): Agent!
  deleteAgent(id: ID!): Boolean!
  deployTopology(id: ID!): Deployment!
  connectAgents(from: ID!, to: ID!, config: ConnectionConfig): Connection!
}

type Subscription {
  agentUpdated(id: ID!): Agent!
  metricsUpdated(agentId: ID!): Metrics!
  topologyChanged: Topology!
}
```

### WebSocket Events

```rust
use visual_editor::{WebSocketHandler, Event};

// Handle WebSocket connections
let ws_handler = WebSocketHandler::new();

// Subscribe to events
ws_handler.on_event(|event| async move {
    match event {
        Event::AgentCreated(agent) => {
            // Broadcast to all clients
        }
        Event::TopologyChanged(topology) => {
            // Update connected clients
        }
        Event::MetricsUpdate(metrics) => {
            // Stream real-time metrics
        }
    }
});
```

### API Endpoints

```rust
use visual_editor::api::{Router, handlers};

// Configure API routes
let router = Router::new()
    .route("/api/graphql", handlers::graphql)
    .route("/api/export/:format", handlers::export)
    .route("/api/import", handlers::import)
    .route("/api/validate", handlers::validate)
    .route("/ws", handlers::websocket);
```

## Frontend Interface

### Component Library

The visual editor includes pre-built components:

- **Agents**: Different types with visual indicators
- **Connections**: Directional arrows with bandwidth indicators
- **Groups**: Logical grouping of related agents
- **Annotations**: Comments and documentation
- **Metrics Overlays**: Real-time performance data

### Topology Operations

```javascript
// Create agent
const agent = editor.createAgent({
  type: 'WebAgent',
  position: { x: 100, y: 100 },
  config: {
    replicas: 3,
    resources: {
      cpu: 1,
      memory: '2Gi'
    }
  }
});

// Connect agents
const connection = editor.connect(agent1, agent2, {
  protocol: 'http',
  port: 8080
});

// Group agents
const group = editor.createGroup([agent1, agent2], {
  name: 'Frontend Services',
  color: '#3498db'
});
```

### Import/Export

```javascript
// Export to Swarm DSL
const dsl = editor.exportToDSL();

// Import from DSL
editor.importFromDSL(dslContent);

// Export to JSON
const json = editor.exportToJSON();

// Export to image
const png = await editor.exportToImage('png');
```

## Real-time Collaboration

```rust
use visual_editor::collaboration::{CollaborationServer, SessionManager};

// Enable real-time collaboration
let collab_server = CollaborationServer::new()
    .max_users_per_session(10)
    .conflict_resolution(ConflictResolution::LastWrite);

// Manage sessions
let session_manager = SessionManager::new();
let session = session_manager.create_session("my-topology")?;

// Handle collaborative edits
collab_server.on_edit(|edit| async move {
    // Validate edit
    if edit.is_valid() {
        // Broadcast to other users
        session.broadcast_edit(edit).await?;
    }
    Ok(())
});
```

## Validation

The editor provides real-time validation:

```rust
use visual_editor::validation::{Validator, ValidationRule};

// Configure validation rules
let validator = Validator::new()
    .add_rule(ValidationRule::MinReplicas(1))
    .add_rule(ValidationRule::MaxConnections(100))
    .add_rule(ValidationRule::RequireHealthCheck)
    .add_rule(ValidationRule::UniqueNames);

// Validate topology
let errors = validator.validate(&topology)?;
if !errors.is_empty() {
    // Show errors in UI
}
```

## Theming

Support for custom themes:

```javascript
// Define custom theme
const customTheme = {
  colors: {
    primary: '#3498db',
    secondary: '#2ecc71',
    background: '#1a1a1a',
    surface: '#2d2d2d',
    error: '#e74c3c'
  },
  fonts: {
    body: 'Inter, sans-serif',
    mono: 'JetBrains Mono, monospace'
  }
};

// Apply theme
editor.applyTheme(customTheme);
```

## Performance

- **Rendering**: 60 FPS with 1000+ nodes
- **WebSocket Latency**: <50ms for updates
- **GraphQL Response**: <100ms for typical queries
- **Memory Usage**: ~50MB for large topologies

## Development

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

### Backend Development

```bash
# Run backend
cargo run

# Run with hot reload
cargo watch -x run

# Run tests
cargo test

# Build release
cargo build --release
```

## Configuration

```toml
[visual_editor]
# Server settings
port = 8080
host = "0.0.0.0"

# GraphQL settings
graphql_path = "/api/graphql"
playground_enabled = true

# WebSocket settings
ws_path = "/ws"
ws_ping_interval = "30s"

# Security
enable_cors = true
allowed_origins = ["http://localhost:3000"]
auth_enabled = false

# Collaboration
collab_enabled = true
max_users_per_session = 10
```

## Testing

```bash
# Backend tests
cargo test

# Frontend tests
cd frontend && npm test

# E2E tests
npm run test:e2e

# Visual regression tests
npm run test:visual
```

## Deployment

### Docker

```dockerfile
# Build frontend
FROM node:18 AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build backend
FROM rust:1.75 AS backend
WORKDIR /app
COPY . .
RUN cargo build --release

# Final image
FROM debian:bookworm-slim
COPY --from=backend /app/target/release/visual-editor /usr/local/bin/
COPY --from=frontend /app/frontend/dist /usr/share/visual-editor/static
EXPOSE 8080
CMD ["visual-editor"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: visual-editor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: visual-editor
  template:
    metadata:
      labels:
        app: visual-editor
    spec:
      containers:
      - name: visual-editor
        image: stratoswarm/visual-editor:latest
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: info
```

## Browser Support

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (WebSocket limitations)
- Mobile: Tablet support (limited mobile phone support)

## License

MIT