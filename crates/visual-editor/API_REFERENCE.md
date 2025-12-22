# StratoSwarm Infrastructure Intelligence Dashboard Backend API

## Overview

The StratoSwarm Infrastructure Intelligence Dashboard Backend provides comprehensive REST and GraphQL APIs for container management, infrastructure monitoring, and system administration. This backend extends the original visual topology editor with full-featured APIs for the React frontend dashboard.

## Quick Start

### Starting the Server
```bash
cargo build --package stratoswarm-visual-editor
./target/debug/visual-editor --port 8080
```

### Health Check
```bash
curl http://localhost:8080/health
```

### Access Web Interface
Navigate to http://localhost:8080 to see the API documentation and available endpoints.

## Core Features

### 1. Container Management APIs

**Templates:**
- `GET /api/containers/templates` - List all container templates
- `POST /api/containers/templates` - Create new template
- `GET /api/containers/templates/{id}` - Get specific template

**Configuration & Validation:**
- `POST /api/containers/validate` - Validate container configuration
- `GET /api/containers/resources` - Get cluster resource availability
- `POST /api/containers/estimate-cost` - Estimate deployment costs

**Library & Deployment:**
- `GET /api/library/templates` - Browse template library
- `GET /api/library/categories` - Get template categories
- `POST /api/library/templates/{id}/deploy` - Deploy from template

### 2. Infrastructure Intelligence APIs

**Topology Management:**
- `GET /api/topology/network` - Network topology visualization
- `GET /api/topology/physical` - Physical infrastructure layout
- `GET /api/topology/logical` - Logical service connections
- `GET /api/topology/performance` - Performance-optimized topology view

**Swarmlet Management:**
- `GET /api/swarmlets` - List all swarmlets with health status
- `GET /api/swarmlets/{id}` - Get individual swarmlet details
- `GET /api/swarmlets/{id}/metrics` - Real-time metrics
- `POST /api/swarmlets/{id}/command` - Execute commands on swarmlet

### 3. System Intelligence APIs

**Monitoring & Alerts:**
- `GET /api/system/overview` - High-level system metrics
- `GET /api/system/recommendations` - AI-driven optimization recommendations  
- `GET /api/system/alerts` - Active system alerts
- `GET /api/system/health` - Overall system health score

**GPU Analytics:**
- `GET /api/gpu/utilization` - Real-time GPU utilization
- `GET /api/gpu/performance` - GPU performance metrics and history
- `GET /api/gpu/allocation` - GPU allocation and scheduling
- `GET /api/gpu/optimization` - GPU optimization recommendations

### 4. Advanced Features APIs

**Configuration Management:**
- `GET /api/config/system` - System-wide configuration
- `GET /api/config/security` - Security configuration
- `GET /api/config/scaling` - Auto-scaling configuration

**Security & Compliance:**
- `GET /api/security/compliance` - Compliance status (GDPR, SOC2, etc.)
- `GET /api/security/vulnerabilities` - Security vulnerability scan results
- `GET /api/security/policies` - Active security policies
- `POST /api/security/scan` - Trigger security scan

**Cost Intelligence:**
- `GET /api/cost/current` - Current resource costs
- `GET /api/cost/forecast` - Cost forecasting and trends  
- `GET /api/cost/optimization` - Cost optimization recommendations
- `GET /api/cost/budgets` - Budget tracking and alerts

### 5. Real-time Features (WebSocket)

The backend provides WebSocket endpoints for live updates:

- `/ws` - Main WebSocket endpoint for all real-time data
  - GPU metrics updates
  - Swarmlet status changes
  - Deployment status updates
  - System alerts and notifications
  - Cost tracking updates
  - Security compliance updates

## API Response Format

All REST endpoints return responses in the following format:

```json
{
  "success": true,
  "data": {...},
  "error": null,
  "timestamp": "2024-01-22T10:30:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "data": null,
  "error": "Error description",
  "timestamp": "2024-01-22T10:30:00Z"
}
```

## GraphQL API

The backend also provides a GraphQL API for complex queries:

- Endpoint: `POST /graphql`
- Playground: `GET /graphql/playground`

## Sample API Calls

### Get Container Templates
```bash
curl http://localhost:8080/api/containers/templates
```

### Validate Container Configuration
```bash
curl -X POST http://localhost:8080/api/containers/validate \
  -H "Content-Type: application/json" \
  -d '{
    "image": "tensorflow/tensorflow:latest-gpu",
    "cpu_cores": 8,
    "memory_gb": 32,
    "gpu_count": 2,
    "gpu_memory_gb": 48
  }'
```

### Get GPU Utilization
```bash
curl http://localhost:8080/api/gpu/utilization
```

### Get System Overview
```bash
curl http://localhost:8080/api/system/overview
```

### Get Current Costs
```bash
curl http://localhost:8080/api/cost/current
```

## Integration Status

- ✅ **REST API Framework** - Complete with Axum
- ✅ **GraphQL Support** - Available with playground
- ✅ **WebSocket Real-time** - All message types implemented
- ✅ **Sample Data** - All endpoints return realistic sample data
- ✅ **Error Handling** - Consistent error responses
- ✅ **API Documentation** - Available at root endpoint

### Pending Integrations

- 🔲 **GPU Agents Integration** - Connect to actual GPU metrics
- 🔲 **Swarmlet Integration** - Real swarmlet management
- 🔲 **Cost Optimization Integration** - Live cost tracking
- 🔲 **Security Scanning** - Actual vulnerability scanning
- 🔲 **Database Layer** - Persistent data storage

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend│────│  Backend API     │────│ Existing Crates │
│   Dashboard     │    │  (visual-editor) │    │ (gpu-agents,    │
└─────────────────┘    └──────────────────┘    │  swarmlet, etc.)│
                                                └─────────────────┘
```

The backend serves as an integration layer between the React dashboard and the existing StratoSwarm crate ecosystem.

## Development

### Adding New Endpoints
1. Add route handler to `rest_api_simple.rs`
2. Update the router in `create_rest_api_router()`
3. Add integration with relevant StratoSwarm crates
4. Update this documentation

### WebSocket Message Types
See `websocket.rs` for all supported real-time message types including:
- Infrastructure metrics
- System alerts
- Cost updates
- Security events
- Performance data

## Performance Considerations

- **Caching**: Implement response caching for expensive operations
- **Rate Limiting**: Add rate limiting for resource-intensive endpoints
- **Pagination**: All list endpoints support pagination parameters
- **Streaming**: Large datasets can be streamed via WebSocket
- **Compression**: Enable response compression for large payloads

## Security Features

- **CORS Enabled**: Cross-origin requests supported
- **Input Validation**: Request validation with detailed error messages
- **Error Sanitization**: No sensitive information in error responses
- **Health Checks**: Built-in health monitoring
- **Structured Logging**: Comprehensive request/response logging

## Monitoring & Observability

The backend includes comprehensive logging and can be monitored using:
- Health checks at `/health`
- Structured logging with tracing
- Performance metrics collection
- Error rate tracking
- Real-time system status via WebSocket

## Contributing

1. Follow Rust best practices and idiomatic code patterns
2. Add comprehensive tests for new endpoints
3. Update API documentation for changes
4. Ensure proper error handling
5. Add real-time WebSocket support for dynamic data

## License

This API backend is part of the StratoSwarm project and follows the same licensing terms.