# Examples, Research, and Experimental Code

This directory contains examples, research notebooks, machine learning models, and experimental code for the UOS Drilling Depth Estimation System.

## 📂 Directory Organization

```
examples/
├── 📁 notebooks/                 # Jupyter notebooks for data analysis
├── 📁 mqtt/                      # MQTT system examples and scripts
├── 📁 machine-learning/          # ML models and training code
├── 📁 validation-studies/        # Algorithm validation and results
├── 📁 data-analysis/             # Specialized data analysis projects
└── 📁 build/                     # Example build configurations
```

## 🎯 Quick Navigation

### 🔬 For Researchers & Data Scientists

| **Need** | **Go Here** | **Description** |
|----------|-------------|-----------------|
| **Jupyter Notebooks** | [`notebooks/`](notebooks/) | Data analysis, visualization, algorithm exploration |
| **ML Models** | [`machine-learning/abyssml/`](machine-learning/abyssml/) | Trained models, inference pipelines |
| **Validation Results** | [`validation-studies/`](validation-studies/) | Algorithm performance analysis |
| **Research Data** | `*/data/` directories | Example datasets and test data |

### 🚀 For Developers & Integration

| **Need** | **Go Here** | **Description** |
|----------|-------------|-----------------|
| **MQTT Examples** | [`mqtt/`](mqtt/) | Live MQTT listeners, publishers, configurations |
| **Build Examples** | [`build/`](build/) | Docker configurations, setup scripts |
| **Integration Code** | [`mqtt/*.py`](mqtt/) | Example integration scripts |

### 📊 For Algorithm Validation

| **Need** | **Go Here** | **Description** |
|----------|-------------|-----------------|
| **Classification Results** | [`validation-studies/validation/Classification/`](validation-studies/validation/Classification/) | Algorithm accuracy analysis |
| **Curve Analysis** | [`validation-studies/validation/Curves/`](validation-studies/validation/Curves/) | Signal processing validation |
| **Performance Data** | [`validation-studies/validation/parquet/`](validation-studies/validation/parquet/) | Processed performance datasets |

## 📋 Directory Details

### 📁 `notebooks/` - Jupyter Notebooks

**Purpose**: Interactive data analysis and algorithm development

**Contents**:
- `HAM_depth_estimation*.ipynb` - Core depth estimation algorithms
- `HAM_ingest_*.ipynb` - Data ingestion and processing
- `est-material-thickness*.ipynb` - Material analysis
- `matrix-profile-explore.ipynb` - Signal analysis techniques
- `setitec-template-extraction*.ipynb` - Template extraction algorithms
- `data/` - Example datasets for notebook analysis

**Getting Started**:
```bash
cd examples/notebooks/
jupyter lab
# Open any .ipynb file to explore
```

### 📁 `mqtt/` - MQTT System Examples

**Purpose**: Real-time MQTT integration examples and tools

**Contents**:
- `listen-continuous.py` - MQTT listener for live data
- `publish-continuous.py` - MQTT publisher for testing
- `listener-simple.py` - Basic MQTT listener example
- `mqtt_conf*.yaml` - Configuration examples
- `MDB_mqtt-depth_estimation.ipynb` - MQTT analysis notebook
- `data/` - Sample MQTT message data

**Getting Started**:
```bash
cd examples/mqtt/

# Run a simple listener
python listener-simple.py

# Run continuous listener with config
python listen-continuous.py --config mqtt_conf.yaml

# Test with publisher
python publish-continuous.py
```

### 📁 `machine-learning/abyssml/` - ML Models

**Purpose**: Machine learning models for depth estimation

**Contents**:
- `src/` - Model implementation and utilities
- `trained_model/` - Pre-trained model artifacts
- `test_data/` - Training and validation datasets
- `README.md` - ML-specific documentation

**Model Types**:
- **has_tool_age_predrilling** - Models accounting for tool age
- **no_tool_age_predrilling** - Models without tool age factor
- Cross-validation folds (cv1-cv12) for robust evaluation

**Getting Started**:
```bash
cd examples/machine-learning/abyssml/

# Run inference
python src/run/inference.py

# C-based inference
python src/run/c_inference.py
```

### 📁 `validation-studies/validation/` - Algorithm Validation

**Purpose**: Comprehensive algorithm performance validation

**Contents**:
- `Classification/` - Algorithm classification results
  - `Algorithm Error/` - Cases where algorithm failed
  - `Good/` - Successful algorithm results
  - `Hard to say/` - Ambiguous cases
  - `Potential Labelling Error/` - Data quality issues
- `Curves/` - Signal analysis and curve fitting validation
- `parquet/` - Processed validation datasets
- `Grip Length Algorithm - Measures on aircraft.pdf` - Reference documentation

**Analysis Categories**:
- **Algorithm Error**: Clear algorithmic failures requiring investigation
- **Good**: Successful depth estimations for reference
- **Hard to say**: Edge cases requiring expert judgment
- **Potential Labelling Error**: Data quality concerns

### 📁 `data-analysis/ti_exit_analysis/` - Specialized Analysis

**Purpose**: Specific analysis projects and reports

**Contents**:
- `reports/` - Analysis reports and processed data
- Tool-specific analysis (Tool 1, Tool 2)
- Performance measurement collections

### 📁 `build/` - Build Examples

**Purpose**: Example build configurations and scripts

**Contents**:
- `Dockerfile.example` - Example Docker build
- `Dockerfile.rpi.publisher` - Raspberry Pi publisher build
- `setup_py.backup` - Python setup configuration backup

## 🛠️ Common Workflows

### Research Workflow
1. **Explore data**: Start with notebooks in `notebooks/`
2. **Validate approach**: Check similar work in `validation-studies/`
3. **Test with MQTT**: Use examples in `mqtt/` for integration
4. **Train models**: Use `machine-learning/abyssml/` for ML development

### Development Workflow
1. **Study examples**: Review relevant code in `mqtt/`
2. **Test locally**: Use `mqtt/` scripts for local testing
3. **Build containers**: Reference `build/` for Docker examples
4. **Validate results**: Compare against `validation-studies/`

### Analysis Workflow
1. **Load data**: Use datasets in various `data/` directories
2. **Run notebooks**: Execute relevant analysis in `notebooks/`
3. **Compare baselines**: Reference `validation-studies/` for benchmarks
4. **Generate reports**: Use `data-analysis/` patterns for reporting

## 📊 Data Guidelines

### Data Formats
- **XLS files**: Original drilling data measurements
- **JSON files**: MQTT message examples and results
- **Parquet files**: Processed datasets for analysis
- **CSV files**: Tabular analysis results
- **PNG files**: Visualization and classification results

### Data Organization
- Each major directory contains its own `data/` subdirectory
- Data files are preserved with original naming for traceability
- Processed data includes clear naming indicating transformations

## 🔧 Configuration Examples

### MQTT Configuration
- `mqtt/mqtt_conf.yaml` - Standard MQTT configuration
- `mqtt/mqtt_docker_conf.yaml` - Docker-specific settings
- `mqtt/mqtt_localhost_conf.yaml` - Local development settings

### Model Configuration
- `machine-learning/abyssml/src/trained_model/*/config.json` - Model configurations
- `machine-learning/abyssml/src/trained_model/*/training_args.bin` - Training parameters

## 🚀 Getting Started

### Prerequisites
```bash
# Install Jupyter (for notebooks)
pip install jupyter jupyterlab

# Install MQTT dependencies (for MQTT examples)
pip install paho-mqtt pyyaml

# Install ML dependencies (for machine learning)
pip install torch transformers pandas numpy
```

### Quick Start Commands
```bash
# Explore notebooks
cd examples/notebooks/ && jupyter lab

# Test MQTT integration
cd examples/mqtt/ && python listener-simple.py

# Run ML inference
cd examples/machine-learning/abyssml/ && python src/run/inference.py

# Review validation results
ls examples/validation-studies/validation/Classification/
```

## 🎓 Learning Path

### 1. **Understanding the System**
- Start with `notebooks/HAM_depth_estimation.ipynb`
- Review `mqtt/MDB_mqtt-depth_estimation.ipynb`
- Examine `validation-studies/` for performance baselines

### 2. **Hands-on Development**
- Run `mqtt/listen-continuous.py` for live data
- Experiment with `notebooks/` for algorithm development
- Use `machine-learning/abyssml/` for ML exploration

### 3. **Advanced Analysis**
- Dive into `validation-studies/` for comprehensive evaluation
- Explore `data-analysis/` for specialized analysis techniques
- Reference `build/` for deployment considerations

## 📚 Additional Resources

- **Main Documentation**: [`../abyss/docs/`](../abyss/docs/)
- **Architecture Guide**: [`../abyss/docs/MQTT_ARCHITECTURE.md`](../abyss/docs/MQTT_ARCHITECTURE.md)
- **Configuration Guide**: [`../abyss/docs/CONFIGURATION_GUIDE.md`](../abyss/docs/CONFIGURATION_GUIDE.md)
- **Developer Guide**: [`../DEVELOPERS.md`](../DEVELOPERS.md)
- **Repository Layout**: [`../REPOSITORY_LAYOUT.md`](../REPOSITORY_LAYOUT.md)

---

**💡 Tip**: Each subdirectory may contain its own README.md with specific details. Always check for local documentation when diving deeper into any area.