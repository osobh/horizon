#!/bin/bash
#
# StratoSwarm Swarmlet Automated Setup Script
# Automates Docker container deployment with cluster token injection
#

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Default values
DEFAULT_IMAGE="stratoswarm/swarmlet:latest"
DEFAULT_CONTAINER_NAME="stratoswarm-swarmlet"
DEFAULT_DATA_DIR="/var/lib/swarmlet"
DEFAULT_CONFIG_DIR="/etc/swarmlet"
DEFAULT_API_PORT="8080"
DEFAULT_METRICS_PORT="9090"

# Script configuration
SCRIPT_VERSION="1.0.0"
INTERACTIVE=false
COMPOSE_MODE=false
CLUSTER_TOKEN=""
CLUSTER_HOST=""
NODE_NAME=""
IMAGE="$DEFAULT_IMAGE"
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
DATA_DIR="$DEFAULT_DATA_DIR"
CONFIG_DIR="$DEFAULT_CONFIG_DIR"
API_PORT="$DEFAULT_API_PORT"
METRICS_PORT="$DEFAULT_METRICS_PORT"
AUTO_START=true
RESTART_POLICY="unless-stopped"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
StratoSwarm Swarmlet Setup Script v${SCRIPT_VERSION}

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -t, --token TOKEN           Cluster admin token (required)
    -c, --cluster HOST[:PORT]   Cluster coordinator address (required)
    -n, --name NAME             Node name (optional, defaults to hostname)
    -i, --interactive           Run in interactive mode
    --compose                   Generate docker-compose.yml instead of running docker
    --image IMAGE               Docker image to use (default: $DEFAULT_IMAGE)
    --container-name NAME       Container name (default: $DEFAULT_CONTAINER_NAME)
    --data-dir PATH             Data directory (default: $DEFAULT_DATA_DIR)
    --config-dir PATH           Config directory (default: $DEFAULT_CONFIG_DIR)
    --api-port PORT             API port (default: $DEFAULT_API_PORT)
    --metrics-port PORT         Metrics port (default: $DEFAULT_METRICS_PORT)
    --no-auto-start             Don't start container automatically
    --restart-policy POLICY     Docker restart policy (default: $RESTART_POLICY)
    -h, --help                  Show this help message
    -v, --version               Show version information

EXAMPLES:
    # Interactive setup
    $0 --interactive

    # Direct setup with token
    $0 --token "abc123" --cluster "cluster.local:7946"

    # Generate docker-compose.yml
    $0 --compose --token "abc123" --cluster "cluster.local:7946"

    # Custom configuration
    $0 --token "abc123" --cluster "192.168.1.100:7946" \\
       --name "edge-node-01" --api-port 8090

For more information, visit: https://docs.stratoswarm.com
EOF
}

# Function to validate requirements
check_requirements() {
    print_status "Checking requirements..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi

    # Check if running as root for certain operations
    if [[ "$DATA_DIR" =~ ^/var || "$CONFIG_DIR" =~ ^/etc ]] && [[ $EUID -ne 0 ]]; then
        print_warning "Running with non-root user. May need sudo for directory creation."
    fi

    print_status "Requirements check passed ✓"
}

# Function to collect information interactively
interactive_setup() {
    print_status "Starting interactive setup..."
    echo

    # Get cluster token
    while [[ -z "$CLUSTER_TOKEN" ]]; do
        echo -n "Enter cluster admin token: "
        read -s CLUSTER_TOKEN
        echo
        if [[ -z "$CLUSTER_TOKEN" ]]; then
            print_error "Token cannot be empty!"
        fi
    done

    # Get cluster host
    while [[ -z "$CLUSTER_HOST" ]]; do
        echo -n "Enter cluster coordinator address (host:port): "
        read CLUSTER_HOST
        if [[ -z "$CLUSTER_HOST" ]]; then
            print_error "Cluster host cannot be empty!"
        elif ! [[ "$CLUSTER_HOST" =~ ^[a-zA-Z0-9.-]+:[0-9]+$ ]]; then
            print_warning "Format should be 'hostname:port' (e.g., 'cluster.local:7946')"
            read -p "Continue anyway? [y/N]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                CLUSTER_HOST=""
            fi
        fi
    done

    # Get node name (optional)
    echo -n "Enter node name [$(hostname)]: "
    read NODE_NAME
    if [[ -z "$NODE_NAME" ]]; then
        NODE_NAME="$(hostname)"
    fi

    # Advanced options
    echo
    read -p "Configure advanced options? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -n "API port [$DEFAULT_API_PORT]: "
        read input_port
        if [[ -n "$input_port" ]]; then
            API_PORT="$input_port"
        fi

        echo -n "Metrics port [$DEFAULT_METRICS_PORT]: "
        read input_port
        if [[ -n "$input_port" ]]; then
            METRICS_PORT="$input_port"
        fi

        echo -n "Data directory [$DEFAULT_DATA_DIR]: "
        read input_dir
        if [[ -n "$input_dir" ]]; then
            DATA_DIR="$input_dir"
        fi

        echo -n "Docker image [$DEFAULT_IMAGE]: "
        read input_image
        if [[ -n "$input_image" ]]; then
            IMAGE="$input_image"
        fi
    fi

    echo
    print_status "Configuration complete!"
}

# Function to validate configuration
validate_config() {
    print_status "Validating configuration..."

    # Check required parameters
    if [[ -z "$CLUSTER_TOKEN" ]]; then
        print_error "Cluster token is required (use --token or --interactive)"
        exit 1
    fi

    if [[ -z "$CLUSTER_HOST" ]]; then
        print_error "Cluster host is required (use --cluster or --interactive)"
        exit 1
    fi

    # Validate ports
    if ! [[ "$API_PORT" =~ ^[0-9]+$ ]] || [[ "$API_PORT" -lt 1 ]] || [[ "$API_PORT" -gt 65535 ]]; then
        print_error "Invalid API port: $API_PORT"
        exit 1
    fi

    if ! [[ "$METRICS_PORT" =~ ^[0-9]+$ ]] || [[ "$METRICS_PORT" -lt 1 ]] || [[ "$METRICS_PORT" -gt 65535 ]]; then
        print_error "Invalid metrics port: $METRICS_PORT"
        exit 1
    fi

    # Check if ports are available
    if ss -tuln | grep -q ":$API_PORT "; then
        print_warning "Port $API_PORT appears to be in use"
    fi

    if ss -tuln | grep -q ":$METRICS_PORT "; then
        print_warning "Port $METRICS_PORT appears to be in use"
    fi

    print_status "Configuration validation passed ✓"
}

# Function to prepare directories
prepare_directories() {
    print_status "Preparing directories..."

    # Create data directory
    if [[ ! -d "$DATA_DIR" ]]; then
        print_debug "Creating data directory: $DATA_DIR"
        mkdir -p "$DATA_DIR" || {
            print_error "Failed to create data directory: $DATA_DIR"
            exit 1
        }
    fi

    # Create config directory
    if [[ ! -d "$CONFIG_DIR" ]]; then
        print_debug "Creating config directory: $CONFIG_DIR"
        mkdir -p "$CONFIG_DIR" || {
            print_error "Failed to create config directory: $CONFIG_DIR"
            exit 1
        }
    fi

    # Set appropriate permissions
    chmod 755 "$DATA_DIR" "$CONFIG_DIR" || {
        print_warning "Failed to set directory permissions"
    }

    print_status "Directories prepared ✓"
}

# Function to generate configuration file
generate_config() {
    local config_file="$CONFIG_DIR/config.toml"
    
    print_status "Generating configuration file: $config_file"

    cat > "$config_file" << EOF
# StratoSwarm Swarmlet Configuration
# Generated by setup script v${SCRIPT_VERSION}
# Generated at: $(date -Iseconds)

[node]
name = "${NODE_NAME:-$(hostname)}"
data_dir = "$DATA_DIR"

[network]
listen = "0.0.0.0:$API_PORT"
discovery = ["mdns", "broadcast"]

[mesh]
join_token = "$CLUSTER_TOKEN"
coordinator = "$CLUSTER_HOST"

[capabilities]
advertise = [
    "workload_execution",
    "command_execution",
    "health_reporting",
    "metrics_collection"
]

[workloads]
accept = ["container", "process", "shell"]

[monitoring]
metrics_enabled = true
metrics_port = $METRICS_PORT
health_check_interval = "30s"

[logging]
level = "info"
file = "$DATA_DIR/swarmlet.log"
max_size = 100
max_files = 5
EOF

    chmod 644 "$config_file" || {
        print_warning "Failed to set config file permissions"
    }

    print_status "Configuration file generated ✓"
}

# Function to generate docker-compose.yml
generate_docker_compose() {
    local compose_file="docker-compose.yml"
    
    print_status "Generating docker-compose.yml..."

    cat > "$compose_file" << EOF
# StratoSwarm Swarmlet Docker Compose
# Generated by setup script v${SCRIPT_VERSION}
# Generated at: $(date -Iseconds)

version: '3.8'

services:
  swarmlet:
    image: $IMAGE
    container_name: $CONTAINER_NAME
    restart: $RESTART_POLICY
    
    environment:
      - SWARMLET_CLUSTER_TOKEN=$CLUSTER_TOKEN
      - SWARMLET_CLUSTER_HOST=$CLUSTER_HOST
      - SWARMLET_NODE_NAME=${NODE_NAME:-$(hostname)}
      - SWARMLET_AUTO_JOIN=true
      - SWARMLET_COMMAND_EXEC=enabled
      - RUST_LOG=info
    
    ports:
      - "$API_PORT:8080"
      - "$METRICS_PORT:9090"
    
    volumes:
      - "$DATA_DIR:/data:rw"
      - "$CONFIG_DIR:/config:ro"
      - "/var/run/docker.sock:/var/run/docker.sock:ro"
    
    networks:
      - swarmlet-net
    
    healthcheck:
      test: ["CMD", "/swarmlet", "profile-hardware"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    
    security_opt:
      - no-new-privileges:true
    
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

networks:
  swarmlet-net:
    driver: bridge
    name: swarmlet-network

volumes:
  swarmlet-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: $DATA_DIR
  
  swarmlet-config:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: $CONFIG_DIR
EOF

    print_status "docker-compose.yml generated ✓"
    print_status "To start the swarmlet:"
    echo "  docker-compose up -d"
    echo
    print_status "To view logs:"
    echo "  docker-compose logs -f"
    echo
    print_status "To stop the swarmlet:"
    echo "  docker-compose down"
}

# Function to pull Docker image
pull_image() {
    print_status "Pulling Docker image: $IMAGE"
    
    if docker pull "$IMAGE"; then
        print_status "Image pulled successfully ✓"
    else
        print_error "Failed to pull image: $IMAGE"
        exit 1
    fi
}

# Function to stop existing container
stop_existing_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_status "Stopping existing container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
        print_status "Existing container removed ✓"
    fi
}

# Function to start Docker container
start_container() {
    print_status "Starting StratoSwarm Swarmlet container..."

    # Build docker run command
    local docker_args=(
        "run"
        "-d"
        "--name" "$CONTAINER_NAME"
        "--restart" "$RESTART_POLICY"
        "-p" "$API_PORT:8080"
        "-p" "$METRICS_PORT:9090"
        "-v" "$DATA_DIR:/data:rw"
        "-v" "$CONFIG_DIR:/config:ro"
        "-v" "/var/run/docker.sock:/var/run/docker.sock:ro"
        "-e" "SWARMLET_CLUSTER_TOKEN=$CLUSTER_TOKEN"
        "-e" "SWARMLET_CLUSTER_HOST=$CLUSTER_HOST"
        "-e" "SWARMLET_NODE_NAME=${NODE_NAME:-$(hostname)}"
        "-e" "SWARMLET_AUTO_JOIN=true"
        "-e" "SWARMLET_COMMAND_EXEC=enabled"
        "-e" "RUST_LOG=info"
        "--security-opt" "no-new-privileges:true"
        "--cap-drop" "ALL"
        "--cap-add" "NET_BIND_SERVICE"
        "$IMAGE"
        "daemon"
        "-c" "/config/config.toml"
    )

    local container_id
    if container_id=$(docker "${docker_args[@]}"); then
        print_status "Container started successfully ✓"
        print_status "Container ID: $container_id"
        
        # Wait a moment for container to initialize
        sleep 2
        
        # Check container status
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            print_status "Container is running ✓"
        else
            print_error "Container failed to start"
            print_error "Container logs:"
            docker logs "$CONTAINER_NAME"
            exit 1
        fi
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to show status
show_status() {
    print_status "Deployment Status:"
    echo
    
    # Container status
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "  Container: ${GREEN}Running ✓${NC}"
    else
        echo -e "  Container: ${RED}Not Running ✗${NC}"
    fi
    
    # API endpoint
    if curl -s "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
        echo -e "  API Health: ${GREEN}Healthy ✓${NC}"
    else
        echo -e "  API Health: ${YELLOW}Checking...${NC}"
    fi
    
    # Metrics endpoint
    if curl -s "http://localhost:$METRICS_PORT/metrics" >/dev/null 2>&1; then
        echo -e "  Metrics: ${GREEN}Available ✓${NC}"
    else
        echo -e "  Metrics: ${YELLOW}Checking...${NC}"
    fi
    
    echo
    print_status "Useful Commands:"
    echo "  View logs:     docker logs -f $CONTAINER_NAME"
    echo "  Stop:          docker stop $CONTAINER_NAME"
    echo "  Restart:       docker restart $CONTAINER_NAME"
    echo "  Shell access:  docker exec -it $CONTAINER_NAME /bin/sh"
    echo
    echo "  API Health:    curl http://localhost:$API_PORT/health"
    echo "  Metrics:       curl http://localhost:$METRICS_PORT/metrics"
    echo
    print_status "Configuration:"
    echo "  Data Dir:      $DATA_DIR"
    echo "  Config Dir:    $CONFIG_DIR"
    echo "  API Port:      $API_PORT"
    echo "  Metrics Port:  $METRICS_PORT"
    echo "  Node Name:     ${NODE_NAME:-$(hostname)}"
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--token)
                CLUSTER_TOKEN="$2"
                shift 2
                ;;
            -c|--cluster)
                CLUSTER_HOST="$2"
                shift 2
                ;;
            -n|--name)
                NODE_NAME="$2"
                shift 2
                ;;
            -i|--interactive)
                INTERACTIVE=true
                shift
                ;;
            --compose)
                COMPOSE_MODE=true
                shift
                ;;
            --image)
                IMAGE="$2"
                shift 2
                ;;
            --container-name)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --config-dir)
                CONFIG_DIR="$2"
                shift 2
                ;;
            --api-port)
                API_PORT="$2"
                shift 2
                ;;
            --metrics-port)
                METRICS_PORT="$2"
                shift 2
                ;;
            --no-auto-start)
                AUTO_START=false
                shift
                ;;
            --restart-policy)
                RESTART_POLICY="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -v|--version)
                echo "StratoSwarm Swarmlet Setup Script v${SCRIPT_VERSION}"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Script header
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║     StratoSwarm Swarmlet Setup         ║"
    echo "║              v${SCRIPT_VERSION}                  ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"

    # Run interactive setup if requested
    if [[ "$INTERACTIVE" == true ]]; then
        interactive_setup
    fi

    # Check requirements
    check_requirements

    # Validate configuration
    validate_config

    # Prepare directories
    prepare_directories

    # Generate configuration
    generate_config

    # Handle compose mode
    if [[ "$COMPOSE_MODE" == true ]]; then
        generate_docker_compose
        exit 0
    fi

    # Pull image if auto-start is enabled
    if [[ "$AUTO_START" == true ]]; then
        pull_image
        stop_existing_container
        start_container
        
        # Show final status
        echo
        show_status
        
        print_status "✅ StratoSwarm Swarmlet setup completed successfully!"
        print_status "Your node should now be connecting to the cluster..."
    else
        print_status "Setup completed. Container not started (--no-auto-start specified)"
        print_status "To start manually:"
        echo "  docker run -d --name $CONTAINER_NAME [... full docker run command ...]"
    fi
}

# Run main function
main "$@"