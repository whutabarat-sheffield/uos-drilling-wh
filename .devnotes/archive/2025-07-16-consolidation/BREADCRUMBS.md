# Development Breadcrumbs

This file tracks what has been requested and the evolution of development priorities.

## User Requests & Context

### Session Context
- **Date**: 2025-07-13
- **Initial Request**: Analyze codebase and create CLAUDE.md guidance
- **Follow-up Request**: Deep analysis and roadmap planning
- **Major Feature Request**: End-user training and retraining platform

### Specific Requests Made

1. **Documentation Creation** ✅ COMPLETED
   - ✅ Created comprehensive DEVNOTES.md in `.devnotes/` directory
   - ✅ Analyzed codebase structure, build commands, and architecture
   - ✅ Documented current state (v0.2.5) with dual sync/async architecture

2. **Deep Analysis & Roadmap Planning** ✅ COMPLETED
   - ✅ **Priority 1**: Move to full Makefile-based installation process (target: v0.3)
   - ✅ **Priority 2**: Migrate to async-only codebase (removing sync components)
   - ✅ **Additional**: Create detailed semantic versioning roadmap for future updates
   - ✅ **Instruction**: "Ultrathink this" - comprehensive analysis completed

3. **Training Platform Feature** ✅ PLANNED
   - ✅ **Hybrid Labeling**: Auto step-code extraction + expert validation GUI
   - ✅ **Dataset Requirements**: 300 labeled + 1000 unlabeled holes (1300 total files)
   - ✅ **Training Approach**: Maintain 24-fold CV, design for future algorithm flexibility
   - ✅ **User Interface**: GUI annotation tool + CLI batch processing
   - ✅ **Target Users**: Basic ML knowledge level
   - ✅ **Architecture**: Dual codebase (PyTorch training + ONNX deployment)

4. **Platform Alternatives Analysis** ✅ COMPLETED
   - ✅ **15 Open Source Platforms Evaluated**: Kubeflow, MLflow, ZenML, ClearML, Flyte, DVC, etc.
   - ✅ **Portainer Compatibility Analysis**: Multi-stack deployment patterns assessed
   - ✅ **Hybrid Recommendation**: 70% open source + 30% custom drilling-specific components
   - ✅ **Cost-Benefit Analysis**: 60% faster development with open source foundation
   - ✅ **Final Recommendation**: MLflow + DVC + Kedro + BentoML + Prefect stack

### Key Insights Discovered
- Current version: 0.2.5 (enhanced duplicate handling and build automation)
- Dual architecture problem: sync (`mqtt/components/`) vs async (`mqtt/async_components/`)
- Build system partially Makefile-based but not comprehensive
- Docker image size issues (~10-13GB) due to PyTorch dependencies
- Legacy code (25+ modules) moved to `abyss/legacy/` in v0.2.4

### Technical Debt Identified
1. **Build System**: Mix of Python scripts, shell scripts, and Makefiles
2. **Architecture Duplication**: Two parallel MQTT processing systems
3. **Dependency Management**: Heavy PyTorch stack for inference-only operations
4. **Configuration Patterns**: Multiple config access methods
5. **Testing Complexity**: Parallel test suites for sync/async

### Next Actions Required
1. **Current State Analysis**: Deep dive into build system, architecture, dependencies
2. **Gap Analysis**: Compare current vs desired state from DEVNOTES
3. **Migration Strategy**: Create detailed roadmap with semantic versioning
4. **Risk Assessment**: Identify breaking changes and mitigation strategies
5. **Implementation Planning**: Phased approach with rollback capabilities

## Future Claude Context
When resuming this work:
1. Review DEVNOTES.md for current system understanding
2. Check this BREADCRUMBS.md for user intentions and progress
3. Focus on Makefile-first approach for v0.3
4. Plan async migration as separate major effort
5. Consider Docker optimization and dependency injection as future priorities

## Development Philosophy
- **Incremental approach**: Avoid massive breaking changes
- **Backward compatibility**: Maintain during transitions
- **Testing-first**: Ensure changes don't break functionality
- **Documentation-driven**: Update docs alongside code changes
- **Performance-conscious**: Consider image size, startup time, runtime efficiency