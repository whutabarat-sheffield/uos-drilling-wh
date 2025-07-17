# UOS Drilling System: Project Overview & Navigation

## Quick Navigation

**New to the project?** Start here:
- 🔧 **Developers** → [SYSTEM_REFERENCE.md](SYSTEM_REFERENCE.md) - Complete architecture and development guide
- 📋 **Project Planning** → [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) - Timeline and resource requirements
- 🎓 **Training Platform** → [TRAINING_PLATFORM_COMPLETE.md](TRAINING_PLATFORM_COMPLETE.md) - Complete training platform documentation (technical + user guide)
- 🏗️ **Architecture Decisions** → [PLATFORM_ARCHITECTURE_DECISIONS.md](PLATFORM_ARCHITECTURE_DECISIONS.md) - Technology choices and implementation decisions

## Project Context

### Current State: v0.2.5
The UOS Drilling System is a deep learning platform for drilling depth estimation that has evolved from a proof-of-concept to a production-ready inference system. The current version (v0.2.5) provides:

- **Dual Architecture**: Sync (`mqtt/components/`) and async (`mqtt/async_components/`) processing
- **Enhanced Duplicate Handling**: Configurable message deduplication with three modes
- **Build Automation**: Comprehensive Docker build scripts and Makefile integration
- **Configuration Management**: Type-safe configuration system with validation
- **24-fold Cross-Validation**: Robust model training and evaluation pipeline

### Development Evolution
The project has undergone significant consolidation and improvement:

**Historical Context:**
- **v0.2.4**: Legacy code consolidation (25+ modules moved to `abyss/legacy/`)
- **v0.2.5**: Enhanced duplicate handling and build automation
- **Current Focus**: Training platform development and architecture optimization

**Technical Debt Addressed:**
- ✅ Configuration patterns unified with ConfigurationManager
- ✅ Build system partially Makefile-based (ongoing improvement)
- ✅ MessageBuffer enhanced with ConfigurationManager integration
- ✅ Docker image optimization (CPU: ~3.5GB, GPU: ~13GB)
- 🔄 Architecture duplication (sync/async) - planned for v0.3+ migration

## Project Vision

### Short-term Goals (v0.3.x)
1. **Complete Makefile Migration**: Unified build system
2. **Async-First Architecture**: Eliminate sync/async duplication
3. **Training Platform MVP**: Basic annotation and retraining capabilities

### Long-term Vision (v1.0+)
Transform from inference-only platform to complete ML training and deployment ecosystem:

- **Hybrid Labeling System**: Auto step-code extraction + expert validation GUI
- **Dataset Management**: Support for 300 labeled + 1000 unlabeled holes
- **User-Friendly Interface**: GUI annotation tool + CLI batch processing
- **Platform Integration**: MLflow + DVC stack
- **Production Deployment**: Optimized for Portainer multi-stack environments

## Document Structure

### Core Documentation (5 Files)

1. **PROJECT_OVERVIEW.md** (this file)
   - Entry point and navigation
   - Project context and vision
   - Quick start guidance

2. **SYSTEM_REFERENCE.md** 
   - Complete technical architecture
   - Development commands and workflows
   - Build system and deployment patterns

3. **DEVELOPMENT_ROADMAP.md**
   - Version progression timeline
   - Resource requirements and estimates
   - Risk assessment and mitigation

4. **TRAINING_PLATFORM_COMPLETE.md**
   - Technical architecture AND user workflows
   - Implementation guide AND usage instructions
   - Developer reference AND troubleshooting
   - Consolidated training platform documentation

5. **PLATFORM_ARCHITECTURE_DECISIONS.md**
   - Technology choices and rationale
   - Implementation patterns and best practices
   - MLflow + DVC architecture decision

### Archive Directory
Historical documents preserved for reference:
- Previous analysis and design documents
- Consolidated content from 17 → 7 → 5 files
- 90% reduction in content duplication

## AI Assistant Integration

The documentation is optimized for AI assistant interrogation with:
- **Self-contained context** for each query
- **Hierarchical structure** for targeted information retrieval
- **Common question → section mapping**
- **Troubleshooting database** with specific solutions
- **Progressive complexity** (basic → advanced workflows)

## Development Philosophy

### Incremental Improvement
- **Avoid massive breaking changes**: Maintain stability during transitions
- **Backward compatibility**: Preserve existing functionality
- **Testing-first approach**: Ensure changes don't break functionality
- **Documentation-driven**: Update docs alongside code changes

### Performance Consciousness
- **Image size optimization**: CPU builds ~3.5GB, targeted improvements
- **Startup time**: Efficient initialization patterns
- **Runtime efficiency**: Async-first architecture planning
- **Resource management**: Memory and CPU optimization

## Getting Started

### For Developers
1. **Environment Setup**: Follow SYSTEM_REFERENCE.md build commands
2. **Architecture Understanding**: Review current dual sync/async system
3. **Development Workflow**: Use provided Docker scripts and Makefile
4. **Testing**: Understand 24-fold CV and validation patterns

### For Project Managers
1. **Timeline Planning**: Consult DEVELOPMENT_ROADMAP.md for estimates
2. **Resource Allocation**: Review training platform requirements
3. **Risk Assessment**: Understand migration challenges and mitigation
4. **Platform Strategy**: Review MLflow + DVC architecture decisions

### For End Users
1. **Training Platform**: Complete workflows in TRAINING_PLATFORM_GUIDE.md
2. **Annotation Tools**: GUI-based labeling and validation
3. **Batch Processing**: CLI tools for large dataset management
4. **Troubleshooting**: Comprehensive solution database

## Maintenance Strategy

This consolidated documentation reduces maintenance overhead by:
- **Single source of truth** for each technical domain
- **Clear ownership** and update responsibilities
- **Reduced duplication** (55% file reduction)
- **Automated validation** through cross-reference checking

## Support and Contribution

For questions, updates, or contributions:
1. **Technical Issues**: Reference SYSTEM_REFERENCE.md
2. **Feature Requests**: Review DEVELOPMENT_ROADMAP.md for planning
3. **Training Platform**: Consult TRAINING_PLATFORM_GUIDE.md
4. **Architecture Decisions**: Review ARCHITECTURE_DECISIONS.md for context

---

*This documentation represents the consolidated knowledge base for the UOS Drilling System, optimized for both human developers and AI assistant integration.*