# Multi-stage build for telemetry-collector

FROM rust:1.75-slim as builder

WORKDIR /build

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY services/telemetry-collector ./services/telemetry-collector

# Build release binary
RUN cargo build --release --package telemetry-collector

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create horizon user
RUN useradd -m -u 1000 horizon

# Create necessary directories
RUN mkdir -p /etc/horizon/certs /var/lib/horizon/telemetry && \
    chown -R horizon:horizon /var/lib/horizon

# Copy binary from builder
COPY --from=builder /build/target/release/telemetry-collector /usr/local/bin/

# Copy example config
COPY services/telemetry-collector/config.example.yaml /etc/horizon/telemetry-collector.yaml

USER horizon

EXPOSE 5001/udp 9091

ENTRYPOINT ["/usr/local/bin/telemetry-collector"]
CMD ["/etc/horizon/telemetry-collector.yaml"]
