#!/bin/bash

# Portainer Stack Updater
# Automatically updates Portainer stacks when changes are detected
# Author: UOS Drilling Team
# Version: 1.0.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check dependencies
check_dependencies() {
    local deps=("curl" "jq" "docker")
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing required dependencies: ${missing[*]}${NC}"
        echo "Please install them first:"
        echo "  Ubuntu/Debian: sudo apt-get install ${missing[*]}"
        echo "  RHEL/CentOS: sudo yum install ${missing[*]}"
        echo "  macOS: brew install ${missing[*]}"
        exit 1
    fi
}

# Load configuration
CONFIG_FILE="${SCRIPT_DIR}/.stack-update-config.env"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    echo "Please create it from the template: cp .stack-update-config.env.example .stack-update-config.env"
    exit 1
fi

# Default values if not set in config
PORTAINER_URL="${PORTAINER_URL:-http://localhost:9000}"
PORTAINER_USER="${PORTAINER_USER:-admin}"
PORTAINER_PASS="${PORTAINER_PASS:-}"
STATE_FILE="${STATE_FILE:-.stack-update-state.json}"
LOG_FILE="${LOG_FILE:-stack-updater.log}"
DRY_RUN="${DRY_RUN:-false}"
AUTO_CONFIRM="${AUTO_CONFIRM:-false}"
ENABLE_NOTIFICATIONS="${ENABLE_NOTIFICATIONS:-false}"

# Lock file for preventing concurrent runs
LOCK_FILE="${SCRIPT_DIR}/.stack-updater.lock"

# JWT token storage
JWT_TOKEN=""
JWT_FILE="${SCRIPT_DIR}/.jwt-token"

# Stack definitions
declare -A STACKS
STACKS["mqtt-broker"]="mqtt-broker/docker-compose.yml"
STACKS["uos-depthest-listener-cpu"]="uos-depthest-listener-cpu/docker-compose.cpu.yml"
STACKS["uos-publisher-json"]="uos-publisher-json/docker-compose.yml"

# Images that need local building
declare -A LOCAL_IMAGES
LOCAL_IMAGES["uos-depthest-listener:cpu"]="build-cpu.sh"
LOCAL_IMAGES["uos-publish-json:latest"]="build-publish.sh"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    rm -f "$LOCK_FILE"
}
trap cleanup EXIT

# Check if another instance is running
check_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
        if ps -p "$pid" > /dev/null 2>&1; then
            log "ERROR" "Another instance is already running (PID: $pid)"
            exit 1
        else
            log "WARN" "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
}

# Authenticate with Portainer
authenticate_portainer() {
    log "INFO" "Authenticating with Portainer..."
    
    # Check if we have a valid token
    if [ -f "$JWT_FILE" ]; then
        JWT_TOKEN=$(cat "$JWT_FILE")
        # Test if token is still valid
        local response=$(curl -k -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${JWT_TOKEN}" \
            "${PORTAINER_URL}/api/users/me")
        
        if [ "$response" = "200" ]; then
            log "INFO" "Using existing valid JWT token"
            return 0
        fi
    fi
    
    # Get new token
    log "DEBUG" "Attempting to authenticate with URL: ${PORTAINER_URL}/api/auth"
    
    local auth_response=$(curl -k -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"${PORTAINER_USER}\",\"password\":\"${PORTAINER_PASS}\"}" \
        "${PORTAINER_URL}/api/auth" 2>&1)
    
    # Extract HTTP status code
    local http_code=$(echo "$auth_response" | tail -n1)
    local response_body=$(echo "$auth_response" | sed '$d')
    
    log "DEBUG" "HTTP Status Code: $http_code"
    
    # Check for common connection errors
    if echo "$response_body" | grep -q "Client sent an HTTP request to an HTTPS server"; then
        log "ERROR" "Protocol mismatch: Portainer is using HTTPS but URL is HTTP"
        log "INFO" "Try changing PORTAINER_URL to use https:// instead of http://"
        log "INFO" "Common Portainer URLs:"
        log "INFO" "  - http://localhost:9000 (non-SSL)"
        log "INFO" "  - https://localhost:9443 (SSL-enabled)"
        return 1
    fi
    
    if echo "$response_body" | grep -q "Connection refused"; then
        log "ERROR" "Cannot connect to Portainer at ${PORTAINER_URL}"
        log "INFO" "Please check:"
        log "INFO" "  - Portainer is running"
        log "INFO" "  - URL and port are correct"
        log "INFO" "  - Firewall is not blocking the connection"
        return 1
    fi
    
    if [ "$http_code" = "000" ]; then
        log "ERROR" "Failed to connect to Portainer at ${PORTAINER_URL}"
        log "INFO" "This usually means:"
        log "INFO" "  - Wrong URL or port"
        log "INFO" "  - Portainer is not running"
        log "INFO" "  - Network/firewall issue"
        return 1
    fi
    
    if [ "$http_code" = "404" ]; then
        log "ERROR" "Portainer API endpoint not found (404)"
        log "INFO" "This might be an older version of Portainer"
        log "INFO" "Check if the API is available at: ${PORTAINER_URL}/api/"
        return 1
    fi
    
    if [ "$http_code" = "401" ] || [ "$http_code" = "403" ]; then
        log "ERROR" "Authentication failed (HTTP $http_code)"
        log "INFO" "Please check your username and password"
        return 1
    fi
    
    JWT_TOKEN=$(echo "$response_body" | jq -r '.jwt // empty' 2>/dev/null || echo "")
    
    if [ -z "$JWT_TOKEN" ]; then
        log "ERROR" "Failed to authenticate with Portainer"
        log "DEBUG" "Full response: $response_body"
        
        # Try to parse error message
        local error_msg=$(echo "$response_body" | jq -r '.message // .details // .error // empty' 2>/dev/null)
        if [ -n "$error_msg" ]; then
            log "ERROR" "Portainer API error: $error_msg"
        else
            log "ERROR" "Unable to parse authentication response"
            log "INFO" "Response body (first 200 chars): ${response_body:0:200}"
        fi
        
        # Additional debugging info
        log "INFO" "Debug information:"
        log "INFO" "  - URL: ${PORTAINER_URL}/api/auth"
        log "INFO" "  - User: ${PORTAINER_USER}"
        log "INFO" "  - Password length: ${#PORTAINER_PASS} characters"
        
        return 1
    fi
    
    echo "$JWT_TOKEN" > "$JWT_FILE"
    chmod 600 "$JWT_FILE"
    log "INFO" "Successfully authenticated with Portainer"
    return 0
}

# Get list of stacks from Portainer
get_portainer_stacks() {
    local response=$(curl -k -s -H "Authorization: Bearer ${JWT_TOKEN}" \
        "${PORTAINER_URL}/api/stacks")
    
    if [ $? -ne 0 ]; then
        log "ERROR" "Failed to get stacks from Portainer"
        return 1
    fi
    
    echo "$response"
}

# Get stack details by name
get_stack_by_name() {
    local stack_name=$1
    local stacks=$(get_portainer_stacks)
    
    echo "$stacks" | jq -r ".[] | select(.Name == \"$stack_name\")"
}

# Check if git has changes for stack files
check_git_changes() {
    local stack_name=$1
    local compose_file=${STACKS[$stack_name]}
    
    if [ ! -f "$compose_file" ]; then
        log "WARN" "Compose file not found: $compose_file"
        return 1
    fi
    
    # Check if file has uncommitted changes
    if git diff --quiet "$compose_file" && git diff --cached --quiet "$compose_file"; then
        # Check if file changed since last update
        local last_update=$(jq -r ".stacks.\"$stack_name\".last_update // 0" "$STATE_FILE" 2>/dev/null || echo "0")
        local file_mtime=$(stat -c %Y "$compose_file" 2>/dev/null || stat -f %m "$compose_file" 2>/dev/null || echo "0")
        
        if [ "$file_mtime" -le "$last_update" ]; then
            return 1 # No changes
        fi
    fi
    
    log "INFO" "Git changes detected for $stack_name"
    return 0
}

# Check if local image needs rebuild
check_image_needs_rebuild() {
    local image=$1
    
    # Check if image is in LOCAL_IMAGES array
    if [[ ! -v LOCAL_IMAGES[$image] ]]; then
        return 1 # Not a local image
    fi
    
    local build_script=${LOCAL_IMAGES[$image]}
    
    # Check if image exists
    if ! docker image inspect "$image" > /dev/null 2>&1; then
        log "INFO" "Image $image does not exist, rebuild needed"
        return 0
    fi
    
    # Check if source files changed since image was built
    local image_created=$(docker image inspect "$image" --format '{{.Created}}' | xargs -I {} date -d {} +%s)
    local latest_source_change=$(find ../abyss -name "*.py" -newer <(date -d @$image_created) -print -quit)
    
    if [ -n "$latest_source_change" ]; then
        log "INFO" "Source files changed since $image was built"
        return 0
    fi
    
    return 1
}

# Rebuild local Docker image
rebuild_local_image() {
    local image=$1
    
    # Check if image is in LOCAL_IMAGES array
    if [[ ! -v LOCAL_IMAGES[$image] ]]; then
        log "ERROR" "No build script found for image: $image"
        return 1
    fi
    
    local build_script=${LOCAL_IMAGES[$image]}
    
    log "INFO" "Rebuilding Docker image: $image"
    
    # Run build script from parent directory
    cd ..
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY RUN] Would run: ./$build_script"
        cd "$SCRIPT_DIR"
        return 0
    fi
    
    if ! ./"$build_script"; then
        log "ERROR" "Failed to build image: $image"
        cd "$SCRIPT_DIR"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    log "INFO" "Successfully rebuilt image: $image"
    return 0
}

# Update stack via Portainer API
update_stack() {
    local stack_name=$1
    local compose_file=${STACKS[$stack_name]}
    
    log "INFO" "Updating stack: $stack_name"
    
    # Get stack details
    local stack=$(get_stack_by_name "$stack_name")
    if [ -z "$stack" ]; then
        log "WARN" "Stack not found in Portainer: $stack_name"
        log "INFO" "To create this stack, use Portainer UI or deploy it first"
        return 1
    fi
    
    local stack_id=$(echo "$stack" | jq -r '.Id')
    local env_id=$(echo "$stack" | jq -r '.EndpointId')
    
    # Read compose file
    local compose_content=$(cat "$compose_file" | jq -Rs .)
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY RUN] Would update stack $stack_name (ID: $stack_id)"
        return 0
    fi
    
    # Update stack
    local update_response=$(curl -k -s -X PUT \
        -H "Authorization: Bearer ${JWT_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"StackFileContent\": $compose_content,
            \"Env\": [],
            \"Prune\": false
        }" \
        "${PORTAINER_URL}/api/stacks/${stack_id}?endpointId=${env_id}")
    
    if echo "$update_response" | jq -e '.message' > /dev/null 2>&1; then
        log "ERROR" "Failed to update stack: $(echo "$update_response" | jq -r '.message')"
        return 1
    fi
    
    log "INFO" "Successfully updated stack: $stack_name"
    
    # Update state file
    update_state "$stack_name" "updated"
    
    return 0
}

# Update state file
update_state() {
    local stack_name=$1
    local status=$2
    local timestamp=$(date +%s)
    
    # Create state file if it doesn't exist
    if [ ! -f "$STATE_FILE" ]; then
        echo '{"stacks":{}}' > "$STATE_FILE"
    fi
    
    # Update state
    local new_state=$(jq ".stacks.\"$stack_name\" = {
        \"last_update\": $timestamp,
        \"status\": \"$status\",
        \"updated_at\": \"$(date -Iseconds)\"
    }" "$STATE_FILE")
    
    echo "$new_state" > "$STATE_FILE"
}

# Send notification
send_notification() {
    local message=$1
    local level=${2:-INFO}
    
    if [ "$ENABLE_NOTIFICATIONS" != "true" ]; then
        return
    fi
    
    # Add notification logic here (webhook, email, etc.)
    log "NOTIFY" "$message"
}

# Health check for stack
check_stack_health() {
    local stack_name=$1
    
    log "INFO" "Checking health of stack: $stack_name"
    
    # Get stack services
    local stack=$(get_stack_by_name "$stack_name")
    if [ -z "$stack" ]; then
        return 1
    fi
    
    # Simple health check - verify all services are running
    # This could be expanded to include actual health endpoints
    sleep 10 # Give services time to start
    
    # Check container status
    local containers=$(docker ps -a --filter "label=com.docker.compose.project=$stack_name" --format "table {{.Names}}\t{{.Status}}\t{{.State}}")
    
    if [ -z "$containers" ]; then
        log "ERROR" "No containers found for stack: $stack_name"
        return 1
    fi
    
    log "DEBUG" "Container status for $stack_name:"
    echo "$containers" | while read line; do
        log "DEBUG" "  $line"
    done
    
    # Check if any container is not running
    local unhealthy=$(docker ps -a --filter "label=com.docker.compose.project=$stack_name" --format "{{.State}}" | grep -v "running" || true)
    
    if [ -n "$unhealthy" ]; then
        log "ERROR" "Some containers in stack $stack_name are not healthy"
        # Show which containers are not running
        docker ps -a --filter "label=com.docker.compose.project=$stack_name" --format "table {{.Names}}\t{{.Status}}" | grep -v "Up" | while read line; do
            log "ERROR" "  $line"
        done
        return 1
    fi
    
    log "INFO" "Stack $stack_name is healthy"
    return 0
}

# Main update process
process_updates() {
    local updates_needed=false
    local failed_updates=()
    
    # Check each stack
    for stack_name in "${!STACKS[@]}"; do
        log "INFO" "Checking stack: $stack_name"
        
        local needs_update=false
        local images_to_rebuild=()
        
        # Check for git changes
        if check_git_changes "$stack_name"; then
            needs_update=true
        fi
        
        # Check if images need rebuild
        local compose_file=${STACKS[$stack_name]}
        local images=$(grep -E "^\s*image:" "$compose_file" | sed 's/.*image:\s*//' | tr -d '"' | tr -d "'")
        
        log "DEBUG" "Found images for $stack_name: $images"
        
        for image in $images; do
            if check_image_needs_rebuild "$image"; then
                needs_update=true
                images_to_rebuild+=("$image")
            fi
        done
        
        if [ "$needs_update" = "true" ]; then
            updates_needed=true
            
            # Rebuild images if needed
            for image in "${images_to_rebuild[@]}"; do
                if ! rebuild_local_image "$image"; then
                    log "ERROR" "Failed to rebuild image $image for stack $stack_name"
                    failed_updates+=("$stack_name")
                    continue 2
                fi
            done
            
            # Update stack
            if ! update_stack "$stack_name"; then
                failed_updates+=("$stack_name")
                continue
            fi
            
            # Health check (skip in dry-run mode since nothing was actually updated)
            if [ "$DRY_RUN" = "true" ]; then
                log "INFO" "[DRY RUN] Would check health after actual update"
            else
                if ! check_stack_health "$stack_name"; then
                    log "ERROR" "Health check failed for stack $stack_name"
                    send_notification "Stack $stack_name failed health check after update" "ERROR"
                    # In a real implementation, we would rollback here
                    failed_updates+=("$stack_name")
                else
                    send_notification "Stack $stack_name updated successfully" "SUCCESS"
                fi
            fi
        else
            log "INFO" "No updates needed for stack: $stack_name"
        fi
    done
    
    # Summary
    if [ "$updates_needed" = "true" ]; then
        if [ ${#failed_updates[@]} -eq 0 ]; then
            log "INFO" "All updates completed successfully"
            send_notification "All stack updates completed successfully" "SUCCESS"
        else
            log "ERROR" "Some updates failed: ${failed_updates[*]}"
            send_notification "Stack updates failed: ${failed_updates[*]}" "ERROR"
            return 1
        fi
    else
        log "INFO" "No updates were needed"
    fi
    
    return 0
}

# Test Portainer connection
test_connection() {
    log "INFO" "Testing Portainer connection..."
    
    # Test basic connectivity
    log "INFO" "Testing URL: ${PORTAINER_URL}"
    local test_response=$(curl -k -s -o /dev/null -w "%{http_code}" "${PORTAINER_URL}" 2>&1)
    
    if [ "$test_response" = "000" ]; then
        log "ERROR" "Cannot reach Portainer at ${PORTAINER_URL}"
        return 1
    fi
    
    log "INFO" "Portainer is reachable (HTTP $test_response)"
    
    # Test API endpoint
    local api_response=$(curl -k -s -o /dev/null -w "%{http_code}" "${PORTAINER_URL}/api" 2>&1)
    log "INFO" "API endpoint check: HTTP $api_response"
    
    # Try to get version info (doesn't require auth)
    local version_info=$(curl -k -s "${PORTAINER_URL}/api/status" 2>&1)
    if echo "$version_info" | jq -e '.Version' > /dev/null 2>&1; then
        local version=$(echo "$version_info" | jq -r '.Version')
        log "INFO" "Portainer version: $version"
    fi
    
    return 0
}

# List stacks in Portainer
list_stacks() {
    log "INFO" "Listing stacks in Portainer..."
    
    local stacks=$(get_portainer_stacks)
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    echo -e "\n${GREEN}Stacks found in Portainer:${NC}"
    echo "$stacks" | jq -r '.[] | "  - \(.Name) (ID: \(.Id), Endpoint: \(.EndpointId))"'
    
    echo -e "\n${YELLOW}Configured stacks in script:${NC}"
    for stack_name in "${!STACKS[@]}"; do
        local stack=$(echo "$stacks" | jq -r ".[] | select(.Name == \"$stack_name\")")
        if [ -n "$stack" ]; then
            echo -e "  ${GREEN}✓${NC} $stack_name - Found in Portainer"
        else
            echo -e "  ${RED}✗${NC} $stack_name - NOT found in Portainer"
        fi
    done
    echo ""
    
    return 0
}

# Main function
main() {
    log "INFO" "Starting Portainer Stack Updater"
    
    # Check dependencies first
    check_dependencies
    
    # Check prerequisites
    check_lock
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                log "INFO" "Running in dry-run mode"
                shift
                ;;
            --force)
                FORCE_UPDATE=true
                log "INFO" "Force update enabled"
                shift
                ;;
            --auto-confirm)
                AUTO_CONFIRM=true
                shift
                ;;
            --test)
                test_connection
                exit $?
                ;;
            --list)
                # Authenticate first
                if ! authenticate_portainer; then
                    log "ERROR" "Failed to authenticate with Portainer"
                    exit 1
                fi
                list_stacks
                exit $?
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --dry-run       Show what would be done without making changes"
                echo "  --force         Force update even if no changes detected"
                echo "  --auto-confirm  Don't prompt for confirmation"
                echo "  --test          Test Portainer connection only"
                echo "  --list          List all stacks in Portainer"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Authenticate with Portainer
    if ! authenticate_portainer; then
        log "ERROR" "Failed to authenticate with Portainer"
        exit 1
    fi
    
    # Process updates
    if ! process_updates; then
        log "ERROR" "Update process failed"
        exit 1
    fi
    
    log "INFO" "Portainer Stack Updater completed"
}

# Run main function
main "$@"