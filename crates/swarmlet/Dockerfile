# Multi-architecture Dockerfile for StratoSwarm Swarmlet
# Builds a minimal container for easy node joining across x86_64 and ARM64

# Build stage
FROM --platform=$BUILDPLATFORM rust:1.79-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    openssl-dev \
    openssl-libs-static

# Set up cross-compilation
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Determine target triple based on platform
RUN case "$TARGETPLATFORM" in \
      "linux/amd64") echo x86_64-unknown-linux-musl > /.target ;; \
      "linux/arm64") echo aarch64-unknown-linux-musl > /.target ;; \
      "linux/arm/v7") echo armv7-unknown-linux-musleabihf > /.target ;; \
      *) echo "Unsupported platform: $TARGETPLATFORM" && exit 1 ;; \
    esac

# Install the target
RUN rustup target add $(cat /.target)

# Set environment variables for cross-compilation
ENV RUSTFLAGS="-C target-feature=+crt-static"
ENV PKG_CONFIG_ALL_STATIC=1
ENV PKG_CONFIG_ALL_DYNAMIC=0

# Create build directory
WORKDIR /build

# Copy workspace files first (for better layer caching)
COPY Cargo.toml Cargo.lock ./
COPY crates/swarmlet/Cargo.toml ./crates/swarmlet/

# Create dummy source files to cache dependencies
RUN mkdir -p crates/swarmlet/src && \
    echo "fn main() {}" > crates/swarmlet/src/main.rs && \
    echo "" > crates/swarmlet/src/lib.rs

# Build dependencies (this layer will be cached unless dependencies change)
RUN cargo build --release --target $(cat /.target) --manifest-path crates/swarmlet/Cargo.toml

# Remove dummy source files
RUN rm -rf crates/swarmlet/src

# Copy actual source code
COPY crates/swarmlet/src ./crates/swarmlet/src
COPY crates/swarmlet/scripts ./crates/swarmlet/scripts

# Build the actual binary
RUN touch crates/swarmlet/src/lib.rs crates/swarmlet/src/main.rs && \
    cargo build --release --target $(cat /.target) --manifest-path crates/swarmlet/Cargo.toml

# Copy the binary to a known location
RUN cp target/$(cat /.target)/release/swarmlet /swarmlet

# Make entrypoint script executable
RUN chmod +x crates/swarmlet/scripts/swarmlet-entrypoint.sh

# Strip the binary to reduce size
RUN strip /swarmlet || true

# Verify the binary works
RUN /swarmlet --help

# Runtime stage - use scratch for minimal size
FROM scratch

# Copy CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the binary
COPY --from=builder /swarmlet /swarmlet

# Copy entrypoint script from build stage
COPY --from=builder /build/crates/swarmlet/scripts/swarmlet-entrypoint.sh /usr/local/bin/entrypoint.sh

# Create minimal filesystem structure
COPY --from=builder /tmp /tmp

# Set up data directory
VOLUME ["/data"]

# Expose default ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/swarmlet", "profile-hardware"]

# Default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["daemon"]

# Metadata
LABEL org.opencontainers.image.title="StratoSwarm Swarmlet"
LABEL org.opencontainers.image.description="Lightweight StratoSwarm node agent for easy cluster joining"
LABEL org.opencontainers.image.url="https://github.com/stratoswarm/stratoswarm"
LABEL org.opencontainers.image.source="https://github.com/stratoswarm/stratoswarm"
LABEL org.opencontainers.image.vendor="StratoSwarm Team"
LABEL org.opencontainers.image.licenses="MIT"