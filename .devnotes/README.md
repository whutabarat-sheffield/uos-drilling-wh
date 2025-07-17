# UOS Drilling System Documentation

This directory contains comprehensive documentation for the UOS Drilling System evolution from v0.2.5 inference-only platform to a complete ML training and deployment ecosystem.

## Core Documentation (5 Authoritative Files)

### **1. PROJECT_OVERVIEW.md**
- Entry point and navigation guide
- Project context and vision
- Quick start for different user types

### **2. SYSTEM_REFERENCE.md**
- Complete system architecture and technical patterns
- Development commands and workflows
- Build system and deployment patterns

### **3. DEVELOPMENT_ROADMAP.md**
- Comprehensive development timeline (v0.3.x → v2.0.x)
- Resource requirements and estimates
- Risk assessment and mitigation strategies

### **4. TRAINING_PLATFORM_COMPLETE.md**
- Complete training platform documentation (technical + user guide)
- PyTorch training + ONNX deployment architecture
- GUI annotation tools and CLI workflows
- Troubleshooting and best practices

### **5. PLATFORM_ARCHITECTURE_DECISIONS.md**
- Technology stack decisions (MLflow + DVC)
- Implementation strategy and rationale
- Rejected alternatives analysis

## Document Purpose & Usage

### For Developers
- Start with **PROJECT_OVERVIEW.md** for navigation and context
- Deep dive into **SYSTEM_REFERENCE.md** for current architecture
- Consult **DEVELOPMENT_ROADMAP.md** for version planning
- Reference **TRAINING_PLATFORM_COMPLETE.md** Part I for implementation details

### For End Users
- **TRAINING_PLATFORM_COMPLETE.md** Part II provides complete user guide
- Optimized for AI assistant queries (Google Gemini, Claude, etc.)
- Includes troubleshooting database and step-by-step workflows

### For Project Managers
- **PROJECT_OVERVIEW.md** for high-level project understanding
- **DEVELOPMENT_ROADMAP.md** contains resource requirements and timeline estimates
- **PLATFORM_ARCHITECTURE_DECISIONS.md** explains technology choices

## Recent Updates

### **MQTT_IMPROVEMENTS_SUMMARY.md** (2025-07-16)
- Comprehensive improvements to MQTT components
- Thread safety enhancements for MessageBuffer
- Logging level optimization (INFO → DEBUG for routine operations)
- Early warning system implementation
- Load testing and robustness improvements

### **OVERVIEW.md**
- High-level system overview moved from root directory
- Contains system architecture and component descriptions

## Archive Directory

The `archive/` directory contains previous documentation versions that were consolidated. Historical consolidations:

### 2025-07-13 Consolidation
- Multiple analysis documents → Initial 7-file structure

### 2025-07-16 Consolidation  
- 15 files → 5 core authoritative documents
- Merged user guide into TRAINING_PLATFORM_COMPLETE.md
- Eliminated redundant meta-documentation
- Created unified PROJECT_OVERVIEW.md entry point

### 2025-07-16 MQTT Improvements
- Moved duplicate analysis files to archive
- Created MQTT_IMPROVEMENTS_SUMMARY.md for consolidated reference
- Organized abyss-specific documentation

## AI Assistant Integration

Documentation is optimized for AI assistant interrogation:
- **Self-contained context** in each document
- **Hierarchical structure** for targeted retrieval
- **User guide section** in TRAINING_PLATFORM_COMPLETE.md Part II
- **Troubleshooting database** with specific solutions
- **Query optimization patterns** for Gemini/Claude

## Maintenance Strategy

This consolidated structure provides:
- **67% reduction** in file count (15 → 5 files)
- **Single source of truth** for each domain
- **Clear ownership** per document
- **Reduced cross-reference complexity**

For updates or questions, start with PROJECT_OVERVIEW.md for navigation.