#!/bin/bash

# Build all Docker images script
# Builds all available Docker configurations with proper error handling

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}    Building All Docker Configurations    ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Track build results
declare -a BUILDS=()
declare -a RESULTS=()
declare -a SIZES=()
declare -a DURATIONS=()

# Function to run a build and track results
run_build() {
    local script_name=$1
    local description=$2
    local start_time=$(date +%s)
    
    echo -e "${YELLOW}Building: $description${NC}"
    echo "Script: $script_name"
    echo ""
    
    if ./"$script_name"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        BUILDS+=("$description")
        RESULTS+=("✅ SUCCESS")
        DURATIONS+=("${duration}s")
        
        # Extract image size (this is a simplified approach)
        local image_line=$(docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "(uos-depthest-listener|uos-publish-json)" | head -1)
        local size=$(echo "$image_line" | awk '{print $3}')
        SIZES+=("$size")
        
        echo -e "${GREEN}✅ $description completed successfully (${duration}s)${NC}"
    else
        BUILDS+=("$description")
        RESULTS+=("❌ FAILED")
        DURATIONS+=("N/A")
        SIZES+=("N/A")
        echo -e "${RED}❌ $description failed${NC}"
    fi
    echo ""
    echo "----------------------------------------"
    echo ""
}

# Check if build scripts exist
echo -e "${YELLOW}Checking build scripts...${NC}"
scripts=("build-main.sh" "build-cpu.sh" "build-runtime.sh" "build-devel.sh" "build-publish.sh")
missing_scripts=()

for script in "${scripts[@]}"; do
    if [ ! -f "$script" ]; then
        missing_scripts+=("$script")
    fi
done

if [ ${#missing_scripts[@]} -ne 0 ]; then
    echo -e "${RED}Error: Missing build scripts:${NC}"
    for script in "${missing_scripts[@]}"; do
        echo "  - $script"
    done
    exit 1
fi

# Make all scripts executable
echo -e "${YELLOW}Making build scripts executable...${NC}"
for script in "${scripts[@]}"; do
    chmod +x "$script"
done
echo ""

# Start building all configurations
total_start_time=$(date +%s)

echo -e "${BLUE}Starting builds...${NC}"
echo ""

# Build 1: Main/Standard build
run_build "build-main.sh" "Main (Standard CPU with PyTorch)"

# Build 2: CPU-only optimized build
run_build "build-cpu.sh" "CPU-Only (Optimized)"

# Build 3: Runtime build with GPU support
run_build "build-runtime.sh" "Runtime (GPU-enabled)"

# Build 4: Development build
run_build "build-devel.sh" "Development (Debug tools)"

# Build 5: Publisher build
run_build "build-publish.sh" "Publisher (JSON data publishing)"

# Calculate total time
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

# Display results summary
echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}           Build Results Summary           ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

printf "%-30s %-12s %-10s %-10s\n" "Build Configuration" "Status" "Duration" "Size"
echo "------------------------------------------------------------"

for i in "${!BUILDS[@]}"; do
    printf "%-30s %-12s %-10s %-10s\n" "${BUILDS[$i]}" "${RESULTS[$i]}" "${DURATIONS[$i]}" "${SIZES[$i]}"
done

echo ""
echo "Total build time: ${total_duration}s"
echo ""

# Count successful and failed builds
successful_builds=$(printf '%s\n' "${RESULTS[@]}" | grep -c "SUCCESS" || true)
failed_builds=$(printf '%s\n' "${RESULTS[@]}" | grep -c "FAILED" || true)

echo -e "${GREEN}Successful builds: $successful_builds${NC}"
if [ $failed_builds -gt 0 ]; then
    echo -e "${RED}Failed builds: $failed_builds${NC}"
fi

echo ""
echo -e "${BLUE}Available Images:${NC}"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | grep -E "(REPOSITORY|uos-depthest-listener|uos-publish-json)"

echo ""
echo -e "${BLUE}===========================================${NC}"

# Exit with appropriate code
if [ $failed_builds -gt 0 ]; then
    echo -e "${RED}Some builds failed. Check the output above for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All builds completed successfully!${NC}"
    exit 0
fi