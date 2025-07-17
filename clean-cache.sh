#!/bin/bash

# Docker cache cleanup script
# Manages Docker build cache to free up disk space when needed

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
FORCE=""
ALL=""
for arg in "$@"; do
    case $arg in
        --force|-f)
            FORCE="yes"
            shift
            ;;
        --all|-a)
            ALL="yes"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --force, -f   Skip confirmation prompts"
            echo "  --all, -a     Remove all images (not just build cache)"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "This script helps manage Docker build cache and images."
            echo "By default, it only cleans build cache. Use --all to remove images too."
            exit 0
            ;;
    esac
done

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}       Docker Cache Cleanup Utility       ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Show current disk usage
echo -e "${YELLOW}Current Docker disk usage:${NC}"
docker system df
echo ""

# Function to get confirmation
confirm() {
    if [ "$FORCE" == "yes" ]; then
        return 0
    fi
    
    read -p "$1 [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Clean build cache
echo -e "${YELLOW}Cleaning Docker build cache...${NC}"
if confirm "Remove Docker build cache?"; then
    echo "Removing build cache..."
    docker builder prune -f
    echo -e "${GREEN}Build cache cleaned!${NC}"
else
    echo "Skipping build cache cleanup."
fi
echo ""

# Clean dangling images
echo -e "${YELLOW}Checking for dangling images...${NC}"
dangling_count=$(docker images -f "dangling=true" -q | wc -l)
if [ $dangling_count -gt 0 ]; then
    echo "Found $dangling_count dangling images."
    if confirm "Remove dangling images?"; then
        docker image prune -f
        echo -e "${GREEN}Dangling images removed!${NC}"
    else
        echo "Skipping dangling image cleanup."
    fi
else
    echo "No dangling images found."
fi
echo ""

# Clean intermediate build stages
echo -e "${YELLOW}Checking for intermediate build stages...${NC}"
intermediate_count=$(docker images --filter label=stage=intermediate -q | wc -l)
if [ $intermediate_count -gt 0 ]; then
    echo "Found $intermediate_count intermediate build stages."
    if confirm "Remove intermediate build stages?"; then
        docker image prune -f --filter label=stage=intermediate
        echo -e "${GREEN}Intermediate stages removed!${NC}"
    else
        echo "Skipping intermediate stage cleanup."
    fi
else
    echo "No intermediate stages found."
fi
echo ""

# Optional: Remove all images
if [ "$ALL" == "yes" ]; then
    echo -e "${RED}WARNING: --all flag specified${NC}"
    echo "This will remove ALL Docker images, not just cache!"
    echo ""
    echo "Current images:"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    echo ""
    
    if confirm "Remove ALL Docker images?"; then
        echo "Removing all images..."
        docker image prune -a -f
        echo -e "${GREEN}All images removed!${NC}"
    else
        echo "Skipping full image cleanup."
    fi
    echo ""
fi

# Show updated disk usage
echo -e "${GREEN}Updated Docker disk usage:${NC}"
docker system df
echo ""

# Provide helpful tips
echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}                   Tips                   ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""
echo "• Use './build-*.sh --no-cache' to force a fresh build"
echo "• Regular builds will now use cache for faster builds"
echo "• Run this script periodically to manage disk space"
echo "• Use 'docker system prune -a' for more aggressive cleanup"
echo ""
echo -e "${GREEN}Cache cleanup completed!${NC}"