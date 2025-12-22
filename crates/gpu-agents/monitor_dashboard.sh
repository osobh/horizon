#!/bin/bash

# Real-time GPU Agents Benchmark Monitoring Dashboard
# Displays live resource usage and progress during benchmark execution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
UPDATE_INTERVAL=1  # seconds
LOG_FILE="benchmark_progress.log"
DASHBOARD_HEIGHT=25
DASHBOARD_WIDTH=80

# Function to clear screen and reset cursor
clear_screen() {
    clear
    tput cup 0 0
}

# Function to draw horizontal line
draw_line() {
    local char=${1:-"─"}
    local width=${2:-$DASHBOARD_WIDTH}
    printf "%*s\n" "$width" | tr ' ' "$char"
}

# Function to get GPU stats
get_gpu_stats() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "0,0,1,0,0"
    else
        echo "0,0,1,0,0"
    fi
}

# Function to get CPU and memory stats
get_system_stats() {
    local cpu_usage=0
    local mem_used=0
    local mem_total=1
    
    # Get CPU usage
    if command -v top &> /dev/null; then
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | sed 's/us,//' | cut -d'.' -f1 || echo "0")
        # Ensure it's a valid integer
        if ! [[ "$cpu_usage" =~ ^[0-9]+$ ]]; then
            cpu_usage=0
        fi
    fi
    
    # Get memory usage
    if [ -f /proc/meminfo ]; then
        mem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        mem_available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        mem_used=$((mem_total - mem_available))
        mem_total_mb=$((mem_total / 1024))
        mem_used_mb=$((mem_used / 1024))
    else
        mem_total_mb=16384
        mem_used_mb=8192
    fi
    
    echo "$cpu_usage,$mem_used_mb,$mem_total_mb"
}

# Function to get benchmark progress from log file
get_benchmark_progress() {
    if [ -f "$LOG_FILE" ]; then
        # Get the latest progress line with better filtering
        local latest_line=$(tail -n 30 "$LOG_FILE" | grep -E "(🚀|🔍|📈|🧠|🕸️|🧬|📊|✅|⏳|💻)" | tail -n 1)
        if [ -n "$latest_line" ]; then
            # Clean up the timestamp format for better display
            echo "$latest_line" | sed 's/\[.*\] //' | cut -c1-76
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S UTC') Monitoring benchmark progress..."
        fi
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S UTC') Waiting for benchmark to start..."
    fi
}

# Function to get current benchmark phase
get_benchmark_phase() {
    if [ -f "$LOG_FILE" ]; then
        local phase_line=$(tail -n 20 "$LOG_FILE" | grep -E "Phase [1-4]/4" | tail -n 1)
        if [ -n "$phase_line" ]; then
            echo "$phase_line" | sed 's/.*Phase /Phase /' | sed 's/\[.*\] //' | cut -c1-40
        else
            echo "Initialization or Setup"
        fi
    else
        echo "Not Started"
    fi
}

# Function to get benchmark completion percentage
get_completion_percentage() {
    if [ -f "$LOG_FILE" ]; then
        # Look for completion percentages in recent log entries
        local percentage=$(tail -n 50 "$LOG_FILE" | grep -oE "[0-9]+\.[0-9]+%|[0-9]+%" | tail -n 1 | sed 's/%//' | cut -d'.' -f1)
        if [ -n "$percentage" ] && [[ "$percentage" =~ ^[0-9]+$ ]]; then
            echo "$percentage"
        else
            # Estimate based on phase if no percentage available
            if grep -q "Phase 1/4" "$LOG_FILE" 2>/dev/null; then
                echo "25"
            elif grep -q "Phase 2/4" "$LOG_FILE" 2>/dev/null; then
                echo "50"
            elif grep -q "Phase 3/4" "$LOG_FILE" 2>/dev/null; then
                echo "75"
            elif grep -q "Phase 4/4" "$LOG_FILE" 2>/dev/null; then
                echo "90"
            elif grep -q "Complete" "$LOG_FILE" 2>/dev/null; then
                echo "100"
            else
                echo "0"
            fi
        fi
    else
        echo "0"
    fi
}

# Function to create a progress bar
create_progress_bar() {
    local percentage=$1
    local width=${2:-40}
    
    # Convert percentage to integer to avoid float arithmetic errors
    local percentage_int=$(echo "$percentage" | cut -d'.' -f1)
    [ -z "$percentage_int" ] && percentage_int=0
    
    local filled=$(( (percentage_int * width) / 100 ))
    local empty=$((width - filled))
    
    printf "["
    printf "%*s" "$filled" | tr ' ' '█'
    printf "%*s" "$empty" | tr ' ' '░'
    printf "]"
}

# Function to format bytes to human readable
format_bytes() {
    local bytes=$1
    if [ "$bytes" -gt 1073741824 ]; then
        echo "$(echo "scale=1; $bytes / 1073741824" | bc)GB"
    elif [ "$bytes" -gt 1048576 ]; then
        echo "$(echo "scale=1; $bytes / 1048576" | bc)MB"
    elif [ "$bytes" -gt 1024 ]; then
        echo "$(echo "scale=1; $bytes / 1024" | bc)KB"
    else
        echo "${bytes}B"
    fi
}

# Function to display the main dashboard
display_dashboard() {
    clear_screen
    
    # Header
    echo -e "${WHITE}┌$(printf '─%.0s' $(seq 1 78))┐${NC}"
    echo -e "${WHITE}│${CYAN}                    🚀 GPU Agents Benchmark Monitor${WHITE}                   │${NC}"
    echo -e "${WHITE}├$(printf '─%.0s' $(seq 1 78))┤${NC}"
    
    # Get current stats and ensure they're integers
    local gpu_stats=$(get_gpu_stats)
    local gpu_util=$(echo "$gpu_stats" | cut -d',' -f1 | cut -d'.' -f1)
    local gpu_mem_used=$(echo "$gpu_stats" | cut -d',' -f2 | cut -d'.' -f1)
    local gpu_mem_total=$(echo "$gpu_stats" | cut -d',' -f3 | cut -d'.' -f1)
    local gpu_temp=$(echo "$gpu_stats" | cut -d',' -f4 | cut -d'.' -f1)
    local gpu_power=$(echo "$gpu_stats" | cut -d',' -f5)
    
    # Ensure all GPU values are valid integers
    [ -z "$gpu_util" ] || ! [[ "$gpu_util" =~ ^[0-9]+$ ]] && gpu_util=0
    [ -z "$gpu_mem_used" ] || ! [[ "$gpu_mem_used" =~ ^[0-9]+$ ]] && gpu_mem_used=0
    [ -z "$gpu_mem_total" ] || ! [[ "$gpu_mem_total" =~ ^[0-9]+$ ]] && gpu_mem_total=1
    [ -z "$gpu_temp" ] || ! [[ "$gpu_temp" =~ ^[0-9]+$ ]] && gpu_temp=0
    
    local system_stats=$(get_system_stats)
    local cpu_util=$(echo "$system_stats" | cut -d',' -f1 | cut -d'.' -f1)
    local mem_used=$(echo "$system_stats" | cut -d',' -f2 | cut -d'.' -f1)
    local mem_total=$(echo "$system_stats" | cut -d',' -f3 | cut -d'.' -f1)
    
    # Ensure all system values are valid integers
    [ -z "$cpu_util" ] || ! [[ "$cpu_util" =~ ^[0-9]+$ ]] && cpu_util=0
    [ -z "$mem_used" ] || ! [[ "$mem_used" =~ ^[0-9]+$ ]] && mem_used=0
    [ -z "$mem_total" ] || ! [[ "$mem_total" =~ ^[0-9]+$ ]] && mem_total=1
    
    # GPU Utilization
    local gpu_util_bar=$(create_progress_bar "$gpu_util" 20)
    echo -e "${WHITE}│${NC} ${GREEN}GPU Utilization:${NC} ${gpu_util_bar} ${gpu_util}%                            ${WHITE}│${NC}"
    
    # GPU Memory (ensure integer calculation)
    local gpu_mem_percent=0
    if [ "$gpu_mem_total" -gt 0 ] 2>/dev/null; then
        gpu_mem_percent=$(( (gpu_mem_used * 100) / gpu_mem_total ))
    fi
    local gpu_mem_bar=$(create_progress_bar "$gpu_mem_percent" 20)
    echo -e "${WHITE}│${NC} ${GREEN}GPU Memory:${NC}      ${gpu_mem_bar} ${gpu_mem_used}MB/${gpu_mem_total}MB              ${WHITE}│${NC}"
    
    # GPU Temperature
    local temp_color="${GREEN}"
    if [ "$gpu_temp" -gt 80 ]; then temp_color="${RED}"; fi
    if [ "$gpu_temp" -gt 70 ]; then temp_color="${YELLOW}"; fi
    echo -e "${WHITE}│${NC} ${GREEN}GPU Temperature:${NC} ${temp_color}${gpu_temp}°C${NC}    Power: ${gpu_power}W                          ${WHITE}│${NC}"
    
    echo -e "${WHITE}├$(printf '─%.0s' $(seq 1 78))┤${NC}"
    
    # CPU Utilization
    local cpu_util_bar=$(create_progress_bar "$cpu_util" 20)
    echo -e "${WHITE}│${NC} ${BLUE}CPU Utilization:${NC} ${cpu_util_bar} ${cpu_util}%                            ${WHITE}│${NC}"
    
    # System Memory (ensure integer calculation)
    local mem_percent=0
    if [ "$mem_total" -gt 0 ] 2>/dev/null; then
        mem_percent=$(( (mem_used * 100) / mem_total ))
    fi
    local mem_bar=$(create_progress_bar "$mem_percent" 20)
    echo -e "${WHITE}│${NC} ${BLUE}System Memory:${NC}   ${mem_bar} ${mem_used}MB/${mem_total}MB              ${WHITE}│${NC}"
    
    echo -e "${WHITE}├$(printf '─%.0s' $(seq 1 78))┤${NC}"
    
    # Benchmark Progress
    local progress_info=$(get_benchmark_progress)
    local current_phase=$(get_benchmark_phase)
    local completion_pct=$(get_completion_percentage)
    
    echo -e "${WHITE}│${NC} ${PURPLE}Current Phase:${NC} $current_phase                                    ${WHITE}│${NC}"
    
    # Show progress bar for overall completion
    local progress_bar=$(create_progress_bar "$completion_pct" 30)
    echo -e "${WHITE}│${NC} ${PURPLE}Completion:${NC}    ${progress_bar} ${completion_pct}%                          ${WHITE}│${NC}"
    
    # Wrap progress info
    local wrapped_progress=$(echo "$progress_info" | fold -w 74)
    while IFS= read -r line; do
        printf "${WHITE}│${NC} %-76s ${WHITE}│${NC}\n" "  $line"
    done <<< "$wrapped_progress"
    
    echo -e "${WHITE}├$(printf '─%.0s' $(seq 1 78))┤${NC}"
    
    # Recent Log Entries
    echo -e "${WHITE}│${NC} ${CYAN}Recent Activity:${NC}                                                    ${WHITE}│${NC}"
    
    if [ -f "$LOG_FILE" ]; then
        local recent_logs=$(tail -n 6 "$LOG_FILE" | while IFS= read -r line; do
            # Extract timestamp and message, focusing on important updates
            if echo "$line" | grep -qE "(🚀|🔍|📈|🧠|🕸️|🧬|📊|✅|⏳|💻|❌|⏰)"; then
                local timestamp=$(echo "$line" | grep -o '\[[^]]*\]' | head -1 | sed 's/\[\(.*\)\]/\1/' | cut -d' ' -f2)
                local message=$(echo "$line" | sed 's/\[[^]]*\] //' | cut -c1-55)
                
                echo "  $timestamp $message"
            fi
        done | tail -n 4)
        
        while IFS= read -r line; do
            printf "${WHITE}│${NC} %-76s ${WHITE}│${NC}\n" "$line"
        done <<< "$recent_logs"
    else
        printf "${WHITE}│${NC} %-76s ${WHITE}│${NC}\n" "  No log file found - benchmark may not be running"
    fi
    
    # Fill remaining space
    local remaining_lines=$((DASHBOARD_HEIGHT - 18))
    for ((i=0; i<remaining_lines; i++)); do
        printf "${WHITE}│${NC} %-76s ${WHITE}│${NC}\n" ""
    done
    
    # Footer
    echo -e "${WHITE}├$(printf '─%.0s' $(seq 1 78))┤${NC}"
    echo -e "${WHITE}│${NC} ${YELLOW}Press Ctrl+C to exit monitoring${NC}    ${CYAN}Log: $LOG_FILE${NC}              ${WHITE}│${NC}"
    echo -e "${WHITE}│${NC} ${YELLOW}Update interval: ${UPDATE_INTERVAL}s${NC}   Benchmark PID: $(pgrep -f 'benchmark' | head -1 || echo 'N/A')                ${WHITE}│${NC}"
    echo -e "${WHITE}└$(printf '─%.0s' $(seq 1 78))┘${NC}"
    
    # Show runtime and update timestamp
    local runtime=$(get_benchmark_runtime)
    echo -e "${CYAN}Runtime: $runtime | Updated: $(date '+%H:%M:%S')${NC}"
}

# Function to check if benchmark is running
is_benchmark_running() {
    pgrep -f "benchmark" > /dev/null 2>&1
}

# Function to get benchmark runtime
get_benchmark_runtime() {
    if [ -f "$LOG_FILE" ]; then
        local start_time=$(head -n 1 "$LOG_FILE" | grep -o '\[[^]]*\]' | sed 's/\[\(.*\)\]/\1/' | head -1)
        if [ -n "$start_time" ]; then
            local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || echo "0")
            local current_epoch=$(date +%s)
            local runtime=$((current_epoch - start_epoch))
            
            if [ $runtime -gt 0 ]; then
                local hours=$((runtime / 3600))
                local minutes=$(((runtime % 3600) / 60))
                local seconds=$((runtime % 60))
                
                if [ $hours -gt 0 ]; then
                    echo "${hours}h ${minutes}m ${seconds}s"
                elif [ $minutes -gt 0 ]; then
                    echo "${minutes}m ${seconds}s"
                else
                    echo "${seconds}s"
                fi
            else
                echo "<1s"
            fi
        else
            echo "Unknown"
        fi
    else
        echo "Not started"
    fi
}

# Function to show usage
show_usage() {
    echo "GPU Agents Benchmark Monitor"
    echo "============================"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -i, --interval SECONDS  Update interval (default: 1)"
    echo "  -l, --logfile FILE      Progress log file (default: benchmark_progress.log)"
    echo "  -h, --help              Show this help"
    echo
    echo "This script provides real-time monitoring of GPU agents benchmark execution."
    echo "Run this in a separate terminal while running benchmarks to see live progress."
    echo
    echo "Example:"
    echo "  Terminal 1: ./benchmark_runner.sh"
    echo "  Terminal 2: ./monitor_dashboard.sh"
}

# Function to cleanup on exit
cleanup() {
    clear_screen
    echo -e "${GREEN}Monitoring stopped.${NC}"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        -l|--logfile)
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate interval
if ! [[ "$UPDATE_INTERVAL" =~ ^[0-9]+$ ]] || [ "$UPDATE_INTERVAL" -lt 1 ]; then
    echo -e "${RED}Error: Update interval must be a positive integer${NC}"
    exit 1
fi

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if required tools are available
if ! command -v bc &> /dev/null; then
    echo -e "${YELLOW}Warning: 'bc' not found. Some calculations may not work properly.${NC}"
fi

# Main monitoring loop
echo -e "${CYAN}Starting GPU Agents Benchmark Monitor...${NC}"
echo -e "${CYAN}Update interval: ${UPDATE_INTERVAL} seconds${NC}"
echo -e "${CYAN}Log file: ${LOG_FILE}${NC}"
echo -e "${CYAN}Press Ctrl+C to exit${NC}"
echo

# Wait a moment for user to read
sleep 2

# Main loop
while true; do
    display_dashboard
    sleep "$UPDATE_INTERVAL"
done