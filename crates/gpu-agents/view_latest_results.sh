#!/bin/bash

# Quick script to view the latest benchmark results

set -e

REPORTS_DIR="reports"
LATEST_LINK="$REPORTS_DIR/latest"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if reports directory exists
if [ ! -d "$REPORTS_DIR" ]; then
    echo -e "${RED}Error: Reports directory '$REPORTS_DIR' not found.${NC}"
    echo "Run benchmarks first: ./benchmark_runner.sh"
    exit 1
fi

# Check if latest symlink exists
if [ ! -L "$LATEST_LINK" ]; then
    echo -e "${RED}Error: No benchmark results found.${NC}"
    echo "Run benchmarks first: ./benchmark_runner.sh"
    exit 1
fi

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo -e "${BLUE}GPU Agents Benchmark Results Viewer${NC}"
    echo "======================================"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  html       Open HTML report in default browser"
    echo "  md         Display markdown report in terminal"
    echo "  json       Show JSON summary with jq (if available)"
    echo "  csv        Display CSV summary"
    echo "  logs       Show benchmark execution logs"
    echo "  system     Display system information"
    echo "  summary    Show quick performance summary"
    echo "  list       List all available benchmark runs"
    echo "  help       Show this help message"
    echo
    echo "Examples:"
    echo "  $0 html      # Open interactive HTML dashboard"
    echo "  $0 summary   # Quick performance overview"
    echo "  $0 json      # Detailed JSON analysis"
    echo
    exit 0
fi

OPTION="$1"
LATEST_DIR=$(readlink -f "$LATEST_LINK")

case "$OPTION" in
    "html")
        HTML_FILE="$LATEST_DIR/benchmark_report.html"
        if [ -f "$HTML_FILE" ]; then
            echo -e "${GREEN}Opening HTML report...${NC}"
            if command -v xdg-open > /dev/null; then
                xdg-open "$HTML_FILE"
            elif command -v open > /dev/null; then
                open "$HTML_FILE"
            else
                echo "HTML report location: $HTML_FILE"
                echo "Open this file in your web browser"
            fi
        else
            echo -e "${RED}Error: HTML report not found${NC}"
            exit 1
        fi
        ;;
    
    "md"|"markdown")
        MD_FILE="$LATEST_DIR/benchmark_report.md"
        if [ -f "$MD_FILE" ]; then
            echo -e "${GREEN}Displaying markdown report...${NC}"
            echo
            if command -v bat > /dev/null; then
                bat "$MD_FILE"
            elif command -v less > /dev/null; then
                less "$MD_FILE"
            else
                cat "$MD_FILE"
            fi
        else
            echo -e "${RED}Error: Markdown report not found${NC}"
            exit 1
        fi
        ;;
    
    "json")
        JSON_FILE="$LATEST_DIR/benchmark_results.json"
        if [ -f "$JSON_FILE" ]; then
            echo -e "${GREEN}Displaying JSON summary...${NC}"
            echo
            if command -v jq > /dev/null; then
                echo -e "${BLUE}System Information:${NC}"
                jq '.system_info' "$JSON_FILE"
                echo
                echo -e "${BLUE}Performance Summary:${NC}"
                jq '.summary' "$JSON_FILE"
            else
                echo "JSON data available at: $JSON_FILE"
                echo "Install 'jq' for formatted JSON viewing"
                echo
                echo "Quick summary:"
                grep -o '"max_agents_spawned":[0-9]*' "$JSON_FILE" | cut -d: -f2 | head -1 | xargs -I {} echo "Max Agents: {}"
                grep -o '"overall_performance_rating":"[^"]*"' "$JSON_FILE" | cut -d: -f2 | tr -d '"' | xargs -I {} echo "Rating: {}"
            fi
        else
            echo -e "${RED}Error: JSON results not found${NC}"
            exit 1
        fi
        ;;
    
    "csv")
        CSV_FILE="$LATEST_DIR/benchmark_summary.csv"
        if [ -f "$CSV_FILE" ]; then
            echo -e "${GREEN}Displaying CSV summary...${NC}"
            echo
            if command -v column > /dev/null; then
                column -t -s, "$CSV_FILE"
            else
                cat "$CSV_FILE"
            fi
        else
            echo -e "${RED}Error: CSV summary not found${NC}"
            exit 1
        fi
        ;;
    
    "logs")
        LOG_FILE="$LATEST_DIR/logs/benchmark.log"
        if [ -f "$LOG_FILE" ]; then
            echo -e "${GREEN}Displaying benchmark logs...${NC}"
            echo
            if command -v less > /dev/null; then
                less "$LOG_FILE"
            else
                cat "$LOG_FILE"
            fi
        else
            echo -e "${RED}Error: Log file not found${NC}"
            exit 1
        fi
        ;;
    
    "system")
        SYSTEM_FILE="$LATEST_DIR/system_info.txt"
        if [ -f "$SYSTEM_FILE" ]; then
            echo -e "${GREEN}Displaying system information...${NC}"
            echo
            cat "$SYSTEM_FILE"
        else
            echo -e "${RED}Error: System info not found${NC}"
            exit 1
        fi
        ;;
    
    "summary")
        JSON_FILE="$LATEST_DIR/benchmark_results.json"
        if [ -f "$JSON_FILE" ]; then
            echo -e "${BLUE}🚀 GPU Agents Benchmark Summary${NC}"
            echo "================================="
            
            # Extract basic info
            local benchmark_dir=$(basename "$LATEST_DIR")
            local timestamp=$(echo "$benchmark_dir" | cut -d_ -f2- | tr '_' ' ')
            echo "📅 Run Date: $timestamp"
            echo "📁 Location: $LATEST_DIR"
            echo
            
            if command -v jq > /dev/null; then
                echo -e "${GREEN}📊 Performance Metrics:${NC}"
                local max_agents=$(jq -r '.summary.max_agents_spawned // "N/A"' "$JSON_FILE")
                local max_llm=$(jq -r '.summary.max_agents_with_llm // "N/A"' "$JSON_FILE")
                local max_kg=$(jq -r '.summary.max_knowledge_graph_nodes // "N/A"' "$JSON_FILE")
                local evolution_score=$(jq -r '.summary.evolution_performance_score // "N/A"' "$JSON_FILE")
                local rating=$(jq -r '.summary.overall_performance_rating // "N/A"' "$JSON_FILE")
                
                echo "• Max Agents Spawned: $max_agents"
                echo "• Max Agents with LLM: $max_llm"
                echo "• Max Knowledge Graph Nodes: $max_kg"
                echo "• Evolution Performance Score: $evolution_score"
                echo "• Overall Rating: $rating"
                
                echo
                echo -e "${GREEN}💻 System Information:${NC}"
                local gpu_name=$(jq -r '.system_info.gpu_name // "N/A"' "$JSON_FILE")
                local gpu_memory=$(jq -r '.system_info.gpu_memory_gb // "N/A"' "$JSON_FILE")
                echo "• GPU: $gpu_name ($gpu_memory GB)"
                
                echo
                echo -e "${GREEN}🎯 Recommendations:${NC}"
                jq -r '.summary.recommendations[]' "$JSON_FILE" 2>/dev/null | sed 's/^/• /' || echo "• No specific recommendations"
            else
                echo "Install 'jq' for detailed summary parsing"
                echo "Raw JSON available at: $JSON_FILE"
            fi
        else
            echo -e "${RED}Error: Benchmark results not found${NC}"
            exit 1
        fi
        ;;
    
    "list")
        echo -e "${BLUE}Available Benchmark Runs:${NC}"
        echo "========================="
        if [ -d "$REPORTS_DIR" ]; then
            local count=0
            for dir in "$REPORTS_DIR"/benchmark_*; do
                if [ -d "$dir" ]; then
                    local basename=$(basename "$dir")
                    local timestamp=$(echo "$basename" | cut -d_ -f2- | tr '_' ' ')
                    local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
                    local is_latest=""
                    if [ "$(readlink -f "$LATEST_LINK")" = "$(readlink -f "$dir")" ]; then
                        is_latest=" ${GREEN}(latest)${NC}"
                    fi
                    echo -e "📁 $basename - $timestamp - $size$is_latest"
                    ((count++))
                fi
            done
            
            if [ $count -eq 0 ]; then
                echo "No benchmark runs found"
            else
                echo
                echo "Total runs: $count"
                echo "Latest: $(readlink "$LATEST_LINK" 2>/dev/null || echo "None")"
            fi
        else
            echo "Reports directory not found"
        fi
        ;;
    
    "help"|"-h"|"--help")
        $0  # Show usage
        ;;
    
    *)
        echo -e "${RED}Error: Unknown option '$OPTION'${NC}"
        echo "Run '$0 help' for available options"
        exit 1
        ;;
esac