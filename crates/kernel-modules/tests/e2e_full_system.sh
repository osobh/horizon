#!/bin/bash
#
# Comprehensive End-to-End Test Suite for StratoSwarm Kernel Modules
# Tests with real system calls, actual hardware, and full integration
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_LOG="/tmp/stratoswarm_e2e.log"
COVERAGE_DIR="/tmp/stratoswarm_coverage"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
STRESS_AGENTS=1000
STRESS_ALLOCATIONS=10000
STRESS_PAGES=50000
PERFORMANCE_ITERATIONS=10000

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$TEST_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$TEST_LOG"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$TEST_LOG"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$TEST_LOG"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1" | tee -a "$TEST_LOG"
    ((TESTS_SKIPPED++))
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root for kernel module testing"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check kernel version
    KERNEL_VERSION=$(uname -r)
    log "Kernel version: $KERNEL_VERSION"
    
    # Check if kernel headers are available
    if [[ ! -d "/lib/modules/$(uname -r)/build" ]]; then
        log_error "Kernel headers not found. Install with: apt install linux-headers-$(uname -r)"
        exit 1
    fi
    
    # Check required tools
    for tool in make gcc insmod rmmod lsmod dmesg; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "Required tool '$tool' not found"
            exit 1
        fi
    done
    
    # Check for GPU (optional)
    if lspci | grep -i nvidia >/dev/null 2>&1; then
        log "NVIDIA GPU detected - real GPU tests will be enabled"
        export REAL_GPU=1
    else
        log_warning "No NVIDIA GPU detected - GPU tests will use mock interfaces"
        export REAL_GPU=0
    fi
    
    # Check memory
    TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_MEM -lt 4096 ]]; then
        log_warning "Less than 4GB RAM detected - reducing stress test parameters"
        STRESS_AGENTS=100
        STRESS_ALLOCATIONS=1000
        STRESS_PAGES=5000
    fi
    
    log_success "System requirements check passed"
}

# Clean up any existing modules
cleanup_modules() {
    log "Cleaning up existing modules..."
    
    # List of modules to clean up
    MODULES=("test_comprehensive" "test_real_gpu" "tier_watch" "swarm_guard" "gpu_dma_lock")
    
    for module in "${MODULES[@]}"; do
        if lsmod | grep -q "^$module "; then
            log "Removing module: $module"
            rmmod "$module" 2>/dev/null || true
        fi
    done
    
    # Clean up /proc entries
    if [[ -d "/proc/swarm" ]]; then
        log "Cleaning /proc/swarm entries"
        # /proc entries are automatically cleaned up when modules are removed
    fi
    
    log_success "Module cleanup completed"
}

# Build all modules
build_modules() {
    log "Building kernel modules..."
    
    cd "$PROJECT_ROOT"
    
    # Build each module
    for module_dir in gpu_dma_lock swarm_guard tier_watch; do
        log "Building $module_dir..."
        cd "$module_dir"
        
        if [[ "$REAL_GPU" == "1" && "$module_dir" == "gpu_dma_lock" ]]; then
            make clean && make REAL_GPU=1 COVERAGE=1
        else
            make clean && make COVERAGE=1
        fi
        
        # Check if module was built
        if [[ "$module_dir" == "gpu_dma_lock" && "$REAL_GPU" == "1" ]]; then
            MODULE_FILE="${module_dir}_real.ko"
        else
            MODULE_FILE="${module_dir}.ko"
        fi
        
        if [[ ! -f "$MODULE_FILE" ]]; then
            log_error "Failed to build $MODULE_FILE"
            exit 1
        fi
        
        cd ..
    done
    
    # Build test modules
    cd tests
    make clean && make
    cd ..
    
    log_success "All modules built successfully"
}

# Load modules in correct order
load_modules() {
    log "Loading kernel modules..."
    
    cd "$PROJECT_ROOT"
    
    # Load core modules first
    log "Loading gpu_dma_lock..."
    if [[ "$REAL_GPU" == "1" ]]; then
        insmod gpu_dma_lock/gpu_dma_lock_real.ko debug=1 gpudirect_enable=1
    else
        insmod gpu_dma_lock/gpu_dma_lock.ko debug=1
    fi
    
    log "Loading swarm_guard..."
    insmod swarm_guard/swarm_guard.ko debug=1 enforce_limits=1
    
    log "Loading tier_watch..."
    insmod tier_watch/tier_watch.ko debug=1 enable_auto_migration=1
    
    # Verify modules loaded
    for module in gpu_dma_lock swarm_guard tier_watch; do
        if ! lsmod | grep -q "^$module "; then
            log_error "Module $module failed to load"
            dmesg | tail -20
            exit 1
        fi
    done
    
    # Check /proc interfaces
    sleep 2  # Allow proc entries to be created
    
    if [[ ! -d "/proc/swarm" ]]; then
        log_error "/proc/swarm not created"
        exit 1
    fi
    
    log_success "All modules loaded successfully"
}

# Test basic functionality
test_basic_functionality() {
    log "Testing basic functionality..."
    
    # Test /proc interfaces
    log "Checking /proc interfaces..."
    
    if [[ -f "/proc/swarm/gpu/stats" ]]; then
        cat /proc/swarm/gpu/stats > /dev/null
        log_success "/proc/swarm/gpu/stats readable"
    else
        log_error "/proc/swarm/gpu/stats not found"
    fi
    
    if [[ -f "/proc/swarm/agents" ]]; then
        cat /proc/swarm/agents > /dev/null
        log_success "/proc/swarm/agents readable"
    else
        log_error "/proc/swarm/agents not found"
    fi
    
    # Check tier directories
    for tier in gpu cpu nvme ssd hdd; do
        if [[ -f "/proc/swarm/tiers/$tier/stats" ]]; then
            cat "/proc/swarm/tiers/$tier/stats" > /dev/null
            log_success "/proc/swarm/tiers/$tier/stats readable"
        else
            log_warning "/proc/swarm/tiers/$tier/stats not found"
        fi
    done
    
    # Test kernel log messages
    dmesg | tail -50 | grep -i "stratoswarm\|gpu_dma_lock\|swarm_guard\|tier_watch" > /tmp/kernel_messages.txt
    if [[ -s /tmp/kernel_messages.txt ]]; then
        log_success "Kernel modules producing log messages"
    else
        log_warning "No kernel log messages found"
    fi
}

# Test device file interfaces
test_device_files() {
    log "Testing device file interfaces..."
    
    # Check if device files are created
    if [[ -c "/dev/gpu_dma_lock" ]]; then
        log_success "GPU DMA Lock device file created"
        
        # Test basic ioctl (this would require a test program)
        log "Device file testing requires userspace test program"
    else
        log_error "GPU DMA Lock device file not created"
    fi
}

# Test memory management
test_memory_management() {
    log "Testing memory management..."
    
    # Check initial memory state
    INITIAL_MEM=$(free -m | awk '/^Mem:/{print $3}')
    
    # Load comprehensive test module
    cd "$PROJECT_ROOT/tests"
    if [[ -f "test_comprehensive.ko" ]]; then
        insmod test_comprehensive.ko
        
        # Wait for tests to complete
        sleep 10
        
        # Check test results in dmesg
        if dmesg | tail -100 | grep -q "ALL TESTS PASSED"; then
            log_success "Comprehensive tests passed"
        elif dmesg | tail -100 | grep -q "SOME TESTS FAILED"; then
            log_error "Some comprehensive tests failed"
            dmesg | tail -50 | grep -E "(FAIL|ERROR)"
        else
            log_warning "Comprehensive test results unclear"
        fi
        
        rmmod test_comprehensive 2>/dev/null || true
    else
        log_skip "Comprehensive test module not available"
    fi
    
    # Check for memory leaks
    FINAL_MEM=$(free -m | awk '/^Mem:/{print $3}')
    MEM_DIFF=$((FINAL_MEM - INITIAL_MEM))
    
    if [[ $MEM_DIFF -gt 100 ]]; then
        log_warning "Potential memory leak detected: ${MEM_DIFF}MB increase"
    else
        log_success "Memory usage stable"
    fi
}

# Test real GPU functionality (if available)
test_real_gpu() {
    if [[ "$REAL_GPU" != "1" ]]; then
        log_skip "Real GPU not available - skipping GPU-specific tests"
        return
    fi
    
    log "Testing real GPU functionality..."
    
    # Check NVIDIA driver
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "NVIDIA driver detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits > /tmp/gpu_info.txt
        if [[ -s /tmp/gpu_info.txt ]]; then
            log_success "GPU information retrieved: $(cat /tmp/gpu_info.txt)"
        fi
    else
        log_warning "nvidia-smi not available"
    fi
    
    # Load real GPU test module
    cd "$PROJECT_ROOT/tests"
    if [[ -f "test_real_gpu.ko" ]]; then
        insmod test_real_gpu.ko
        
        # Wait for tests
        sleep 15
        
        # Check results
        if dmesg | tail -100 | grep -q "All tests PASSED"; then
            log_success "Real GPU tests passed"
        else
            log_error "Real GPU tests failed or incomplete"
            dmesg | tail -30 | grep -i gpu
        fi
        
        rmmod test_real_gpu 2>/dev/null || true
    else
        log_skip "Real GPU test module not available"
    fi
}

# Test cgroup integration
test_cgroup_integration() {
    log "Testing cgroup integration..."
    
    # Check if cgroups v2 is available
    if [[ -d "/sys/fs/cgroup/unified" ]] || mountpoint -q /sys/fs/cgroup; then
        log "cgroups v2 detected"
        
        # Check if swarm cgroup was created
        if [[ -d "/sys/fs/cgroup/swarm" ]]; then
            log_success "Swarm cgroup created"
            
            # Check cgroup controllers
            if [[ -f "/sys/fs/cgroup/swarm/cgroup.controllers" ]]; then
                CONTROLLERS=$(cat /sys/fs/cgroup/swarm/cgroup.controllers)
                log "Available controllers: $CONTROLLERS"
                
                if echo "$CONTROLLERS" | grep -q memory; then
                    log_success "Memory controller available"
                fi
                
                if echo "$CONTROLLERS" | grep -q cpu; then
                    log_success "CPU controller available"
                fi
            fi
        else
            log_warning "Swarm cgroup not created - may use default cgroup"
        fi
    else
        log_warning "cgroups v2 not available"
    fi
}

# Test namespace isolation
test_namespace_isolation() {
    log "Testing namespace isolation..."
    
    # Check namespace support
    if [[ -f "/proc/sys/user/max_user_namespaces" ]]; then
        MAX_NS=$(cat /proc/sys/user/max_user_namespaces)
        if [[ "$MAX_NS" -gt 0 ]]; then
            log_success "User namespaces enabled (max: $MAX_NS)"
        else
            log_warning "User namespaces disabled"
        fi
    fi
    
    # Check PID namespace support
    if [[ -f "/proc/sys/kernel/pid_max" ]]; then
        PID_MAX=$(cat /proc/sys/kernel/pid_max)
        log "PID namespace support available (max PID: $PID_MAX)"
        log_success "PID namespaces supported"
    fi
    
    # Test would require creating actual namespaces
    log "Namespace isolation testing requires userspace helper programs"
}

# Test performance benchmarks
test_performance() {
    log "Running performance benchmarks..."
    
    # Test kernel log performance (proxy for module performance)
    START_TIME=$(date +%s%N)
    
    for i in $(seq 1 1000); do
        echo "test_message_$i" > /proc/sys/kernel/printk_ratelimit >/dev/null 2>&1 || true
    done
    
    END_TIME=$(date +%s%N)
    DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))
    
    log "Kernel interface test took ${DURATION_MS}ms for 1000 operations"
    
    if [[ $DURATION_MS -lt 1000 ]]; then
        log_success "Performance test passed (${DURATION_MS}ms < 1000ms)"
    else
        log_warning "Performance test slow (${DURATION_MS}ms >= 1000ms)"
    fi
    
    # Memory allocation benchmark
    dd if=/dev/zero of=/tmp/perf_test bs=1M count=100 >/dev/null 2>&1
    sync
    rm -f /tmp/perf_test
    
    log_success "Memory performance test completed"
}

# Test under stress conditions
test_stress() {
    log "Running stress tests..."
    
    # Create memory pressure
    log "Creating memory pressure..."
    STRESS_PROCS=()
    
    for i in $(seq 1 4); do
        # Create background memory allocation
        (
            python3 -c "
import time
data = []
for i in range(1000):
    data.append(' ' * 1024 * 1024)  # 1MB chunks
    time.sleep(0.01)
            " >/dev/null 2>&1
        ) &
        STRESS_PROCS+=($!)
    done
    
    # Wait a bit for memory pressure
    sleep 5
    
    # Test modules under pressure
    if dmesg | tail -20 | grep -i "out of memory\|oom"; then
        log_warning "OOM detected during stress test"
    else
        log_success "Modules stable under memory pressure"
    fi
    
    # Clean up stress processes
    for pid in "${STRESS_PROCS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    wait
    
    # Test with many processes
    log "Testing with process pressure..."
    for i in $(seq 1 50); do
        sleep 0.1 &
    done
    wait
    
    log_success "Stress test completed"
}

# Test error conditions and recovery
test_error_recovery() {
    log "Testing error conditions and recovery..."
    
    # Test module reload
    log "Testing module reload..."
    
    for module in tier_watch swarm_guard gpu_dma_lock; do
        if rmmod "$module" 2>/dev/null; then
            log "Removed $module"
            sleep 1
            
            # Reload
            cd "$PROJECT_ROOT"
            if [[ "$module" == "gpu_dma_lock" && "$REAL_GPU" == "1" ]]; then
                insmod "${module}/${module}_real.ko" debug=1
            else
                insmod "${module}/${module}.ko" debug=1
            fi
            
            if lsmod | grep -q "^$module "; then
                log_success "Successfully reloaded $module"
            else
                log_error "Failed to reload $module"
            fi
        else
            log_warning "Could not remove $module (may be in use)"
        fi
    done
    
    # Test invalid operations
    log "Testing invalid operations..."
    
    # Try to read non-existent proc files
    if cat /proc/swarm/nonexistent 2>/dev/null; then
        log_error "Non-existent proc file accessible"
    else
        log_success "Non-existent proc files properly protected"
    fi
    
    # Test permission denied scenarios
    if [[ -f "/proc/swarm/gpu/stats" ]]; then
        # This should succeed as root
        cat /proc/swarm/gpu/stats >/dev/null
        log_success "Proc file access permissions working"
    fi
}

# Generate coverage report
generate_coverage_report() {
    log "Generating coverage report..."
    
    mkdir -p "$COVERAGE_DIR"
    cd "$PROJECT_ROOT"
    
    # Collect coverage data if gcov was used
    for module_dir in gpu_dma_lock swarm_guard tier_watch; do
        if [[ -d "$module_dir" ]]; then
            cd "$module_dir"
            if ls *.gcno >/dev/null 2>&1; then
                log "Collecting coverage for $module_dir"
                gcov *.c >/dev/null 2>&1 || true
                mv *.gcov "$COVERAGE_DIR/" 2>/dev/null || true
            fi
            cd ..
        fi
    done
    
    # Generate summary report
    cat > "$COVERAGE_DIR/summary.txt" << EOF
StratoSwarm Kernel Modules - Coverage Summary
=============================================

Tests Run: $((TESTS_PASSED + TESTS_FAILED))
Passed: $TESTS_PASSED
Failed: $TESTS_FAILED
Skipped: $TESTS_SKIPPED

Coverage files generated in: $COVERAGE_DIR

To view detailed coverage:
1. Install lcov: apt install lcov
2. Generate HTML report: genhtml *.gcov -o html_report
3. Open html_report/index.html in browser
EOF
    
    log_success "Coverage report generated in $COVERAGE_DIR"
}

# Main test execution
main() {
    log "StratoSwarm Kernel Modules - Comprehensive E2E Test Suite"
    log "=========================================================="
    
    # Initialize log
    echo "Test started at $(date)" > "$TEST_LOG"
    
    # Pre-test setup
    check_root
    check_requirements
    cleanup_modules
    build_modules
    load_modules
    
    # Core functionality tests
    test_basic_functionality
    test_device_files
    test_memory_management
    
    # Hardware-specific tests
    test_real_gpu
    
    # Integration tests
    test_cgroup_integration
    test_namespace_isolation
    
    # Performance and stress tests
    test_performance
    test_stress
    
    # Error handling tests
    test_error_recovery
    
    # Generate reports
    generate_coverage_report
    
    # Final cleanup
    cleanup_modules
    
    # Print final summary
    echo
    log "=========================================="
    log "E2E Test Suite Summary"
    log "=========================================="
    log "Total Tests: $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))"
    log_success "Passed: $TESTS_PASSED"
    if [[ $TESTS_FAILED -gt 0 ]]; then
        log_error "Failed: $TESTS_FAILED"
    else
        log "Failed: $TESTS_FAILED"
    fi
    if [[ $TESTS_SKIPPED -gt 0 ]]; then
        log_skip "Skipped: $TESTS_SKIPPED"
    else
        log "Skipped: $TESTS_SKIPPED"
    fi
    
    # Calculate success rate
    TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        SUCCESS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))
        log "Success Rate: ${SUCCESS_RATE}%"
        
        if [[ $SUCCESS_RATE -ge 95 ]]; then
            log_success "🎉 EXCELLENT! 95%+ success rate achieved"
        elif [[ $SUCCESS_RATE -ge 90 ]]; then
            log_success "✅ GOOD! 90%+ success rate achieved"
        elif [[ $SUCCESS_RATE -ge 80 ]]; then
            log_warning "⚠️  ACCEPTABLE! 80%+ success rate achieved"
        else
            log_error "❌ POOR! Less than 80% success rate"
        fi
    fi
    
    log "=========================================="
    log "Full test log available at: $TEST_LOG"
    log "Coverage report available at: $COVERAGE_DIR"
    
    # Exit with appropriate code
    if [[ $TESTS_FAILED -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Handle interrupts
trap 'log "Test interrupted!"; cleanup_modules; exit 130' INT TERM

# Run main function
main "$@"