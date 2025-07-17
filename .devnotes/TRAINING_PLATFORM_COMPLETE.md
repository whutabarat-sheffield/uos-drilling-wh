# UOS Drilling Training Platform: Complete Architecture, Implementation & User Guide

## Platform Overview

The UOS Drilling Training Platform enables end-users to perform their own training and retraining of deep learning models for drilling depth estimation. Built with a **dual codebase architecture** (PyTorch training + ONNX deployment), the platform supports hybrid labeling workflows, GUI annotation tools, and algorithm-agnostic design for future extensibility.

**Target Users**: Industrial drilling professionals with basic ML knowledge
**Dataset Scale**: 1300 drilling files (300 labeled + 1000 unlabeled holes)
**Training Approach**: Current 24-fold CV with future algorithm flexibility

> **Note**: This document consolidates both technical implementation details (Part I) and user guide content (Part II) into a single comprehensive reference.

## Core Requirements & User Workflows

### Training Dataset Requirements

**Dataset Composition**:
- **300 labeled holes**: Expert-validated drilling operations with step-code annotations
- **1000 unlabeled holes**: Pre-training data for semi-supervised learning
- **Material Coverage**: Representative samples across different drilling configurations
- **Quality Standards**: 100+ drilling operations per material/configuration type

**Data Quality Criteria**:
- Valid step transitions (1→2→N) for training labels
- No missing values in critical channels (Position, Torque, Thrust)
- Consistent sampling rates across files
- Comprehensive metadata (material type, drilling parameters)

### User Workflow Patterns

#### **Guided Workflow (Basic ML Users)**
```
1. Data Upload → 2. Automated Pre-labeling → 3. GUI Validation → 4. Training → 5. Model Export
```

#### **Advanced Workflow (Experienced Users)**  
```
1. Batch Data Processing → 2. CLI Configuration → 3. Custom Training → 4. Performance Analysis → 5. Production Deployment
```

## Data Structure & Processing Architecture

### Setitec XLS File Format

**Primary Input Format**: Setitec XLS files (tab-separated text with .xls extension)

**Core Data Channels**:
```python
# Essential drilling signals
core_channels = {
    'Position (mm)': 'drilling_depth_mm',
    'I Torque (A)': 'torque_current_amps', 
    'I Thrust (A)': 'thrust_current_amps',
    'Step (nb)': 'drilling_step_codes'
}
```

**Feature Engineering Pipeline**:
```python
# Complete feature vector (17 channels)
def create_feature_vector(setitec_data):
    return np.array([
        # Position features
        position_raw,           # Raw position signal
        position_velocity,      # First derivative
        position_acceleration,  # Second derivative
        
        # Torque features  
        torque_raw,            # Raw torque signal
        torque_smoothed,       # Filtered signal
        torque_gradient,       # Rate of change
        
        # Thrust features
        thrust_raw,            # Raw thrust signal  
        thrust_smoothed,       # Filtered signal
        thrust_gradient,       # Rate of change
        
        # Combined features
        torque_thrust_ratio,   # Torque/thrust interaction
        power_estimate,        # Estimated power consumption
        step_code_binary,      # Step transition indicators
        
        # Quality indicators
        signal_confidence,     # Data quality score
        noise_level,          # Signal-to-noise ratio
        sampling_consistency,  # Temporal consistency
        
        # Derived metrics
        drilling_efficiency,   # Performance indicator
        wear_estimation       # Tool wear proxy
    ])
```

### Data Validation Framework

**Automated Quality Checks**:
```python
class DataQualityValidator:
    """Comprehensive validation for drilling files"""
    
    def validate_file_structure(self, file_path):
        """Validate Setitec XLS structure and format"""
        
    def validate_signal_quality(self, signals):
        """Check for missing values, outliers, noise"""
        
    def validate_step_sequences(self, step_codes):
        """Ensure valid drilling step transitions"""
        
    def validate_temporal_consistency(self, timestamps):
        """Check sampling rate and temporal gaps"""
        
    def generate_quality_report(self, validation_results):
        """Comprehensive quality assessment report"""
```

## Hybrid Labeling Strategy

### Automated Label Generation

**Step-Code Based Extraction**:
```python
class AutomaticLabeler:
    """Extract depth labels from step codes"""
    
    def extract_depth_points(self, step_codes, position_signal):
        """
        Identify drilling phases from step transitions:
        - Entry point: Step 1 → Step 2 transition
        - Transition point: Step 2 → Step N transition  
        - Exit point: Final step completion
        """
        return {
            'entry_depth': self._find_step_transition(1, 2),
            'transition_depth': self._find_step_transition(2, 'any'),
            'exit_depth': self._find_final_step()
        }
    
    def calculate_confidence(self, labels, signal_quality):
        """Estimate label reliability (high/medium/low)"""
        return confidence_score  # 0.0-1.0 reliability metric
```

**Label Confidence Categories**:
- **High Confidence (>0.8)**: Clear step transitions, good signal quality
- **Medium Confidence (0.5-0.8)**: Acceptable transitions, minor signal issues
- **Low Confidence (<0.5)**: Unclear transitions, requires expert validation

### Expert Validation Workflow

**GUI Validation Process**:
1. **Auto-labels loaded** with confidence indicators
2. **Expert review** of medium/low confidence labels
3. **Interactive adjustment** of depth points via drag-and-drop
4. **Quality validation** with real-time feedback
5. **Batch approval** for high-confidence labels

## Algorithm-Agnostic Training Architecture

### Abstract Training Strategy Interface

```python
from abc import ABC, abstractmethod

class TrainingStrategy(ABC):
    """Abstract base for all training algorithms"""
    
    @abstractmethod
    def prepare_data(self, raw_data, labels):
        """Convert raw data to algorithm-specific format"""
        pass
    
    @abstractmethod  
    def train_model(self, training_data, config):
        """Execute training with given configuration"""
        pass
    
    @abstractmethod
    def evaluate_model(self, model, test_data):
        """Perform model evaluation and metrics calculation"""
        pass
    
    @abstractmethod
    def export_model(self, model, export_format='onnx'):
        """Export trained model for deployment"""
        pass
```

### Current Implementation: PatchTSMixer Strategy

```python
class PatchTSMixerStrategy(TrainingStrategy):
    """Current 24-fold cross-validation implementation"""
    
    def __init__(self):
        self.cv_folds = 24
        self.sequence_length = 512
        self.feature_channels = 17
        
    def prepare_data(self, raw_data, labels):
        """
        Convert Setitec signals to PatchTSMixer format:
        - Windowed sequences (512 samples)
        - 17-channel feature vectors
        - 24-fold cross-validation splits
        """
        return self._create_cv_splits(windowed_data)
    
    def train_model(self, training_data, config):
        """Execute 24-fold CV training with PyTorch"""
        models = []
        for fold_idx in range(self.cv_folds):
            fold_model = self._train_fold(training_data[fold_idx])
            models.append(fold_model)
        return EnsembleModel(models)
```

### Future Training Strategies

**Extensible Algorithm Registry**:
```python
class ModelRegistry:
    """Registry for all supported training algorithms"""
    
    strategies = {
        # Current implementation
        'patchtsmixer_24cv': PatchTSMixerStrategy,
        
        # Future algorithms
        'transformer_attention': TransformerStrategy,
        'cnn_lstm_hybrid': CNNLSTMStrategy, 
        'ensemble_voting': EnsembleStrategy,
        'neural_ode_continuous': NeuralODEStrategy
    }
    
    @classmethod
    def get_strategy(cls, strategy_name):
        """Factory method for training strategy selection"""
        return cls.strategies[strategy_name]()
```

## Training Platform Architecture

### System Component Structure

```
training_platform/
├── data/                           # Data management layer
│   ├── ingestion/                 # Setitec XLS file processing
│   ├── validation/                # Quality assurance framework
│   ├── labeling/                  # Hybrid labeling system
│   └── preprocessing/             # Feature engineering pipeline
├── training/                      # Training orchestration layer
│   ├── strategies/                # Algorithm implementations
│   ├── orchestrator/              # Training pipeline management
│   ├── hyperparameters/           # Parameter optimization
│   └── evaluation/                # Model assessment
├── models/                        # Model management layer
│   ├── registry/                  # Model versioning (MLflow)
│   ├── conversion/                # PyTorch → ONNX pipeline
│   ├── validation/                # Model performance testing
│   └── deployment/                # Production model serving
├── interfaces/                    # User interface layer
│   ├── gui/                       # PyQt6 annotation tool
│   ├── cli/                       # Command-line workflows
│   ├── web/                       # Optional web interface
│   └── api/                       # REST API for integration
└── infrastructure/                # Platform infrastructure
    ├── experiment_tracking/       # MLflow integration
    ├── data_versioning/           # DVC integration
    ├── monitoring/                # Training job monitoring
    └── configuration/             # Unified config management
```

### Training Orchestration Engine

```python
class TrainingOrchestrator:
    """Complete training pipeline management"""
    
    def __init__(self, config_path, strategy_name='patchtsmixer_24cv'):
        self.config = ConfigurationManager(config_path)
        self.strategy = ModelRegistry.get_strategy(strategy_name)
        self.experiment_tracker = MLflowTracker()
        
    def execute_full_pipeline(self, data_directory):
        """End-to-end training workflow"""
        
        # Phase 1: Data Pipeline
        raw_data = self._ingest_setitec_files(data_directory)
        validated_data = self._validate_data_quality(raw_data)
        
        # Phase 2: Labeling Pipeline  
        auto_labels = self._generate_automatic_labels(validated_data)
        expert_labels = self._collect_expert_validations(auto_labels)
        final_labels = self._merge_labeling_sources(auto_labels, expert_labels)
        
        # Phase 3: Training Pipeline
        prepared_data = self.strategy.prepare_data(validated_data, final_labels)
        trained_model = self.strategy.train_model(prepared_data, self.config)
        
        # Phase 4: Evaluation Pipeline
        performance_metrics = self.strategy.evaluate_model(trained_model, test_data)
        
        # Phase 5: Deployment Pipeline
        onnx_model = self.strategy.export_model(trained_model, 'onnx')
        deployment_package = self._create_deployment_package(onnx_model)
        
        # Phase 6: Experiment Tracking
        self.experiment_tracker.log_experiment(
            strategy=strategy_name,
            data_version=data_directory,
            model_performance=performance_metrics,
            model_artifacts=deployment_package
        )
        
        return deployment_package
```

## GUI Annotation Tool Architecture

### PyQt6 Interface Design

**Main Application Window**:
```python
class DrillingSuggestionAnnotationTool(QMainWindow):
    """Primary GUI for drilling data annotation"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_signal_connections()
        
    def setup_ui(self):
        """Create three-panel layout"""
        
        # Left Panel: File Management
        self.file_browser = FileBrowserWidget()
        self.annotation_queue = AnnotationQueueWidget()
        
        # Center Panel: Signal Visualization
        self.signal_plotter = InteractiveSignalPlotter()
        self.annotation_overlay = DepthAnnotationOverlay()
        
        # Right Panel: Validation & Metrics
        self.label_validator = LabelValidationWidget()
        self.quality_metrics = QualityMetricsWidget()
```

**Interactive Signal Visualization**:
```python
class InteractiveSignalPlotter(QWidget):
    """Multi-signal plotting with annotation capabilities"""
    
    def __init__(self):
        self.signals = ['position', 'torque', 'thrust', 'step_codes']
        self.setup_matplotlib_canvas()
        
    def plot_drilling_signals(self, file_data):
        """Display all drilling signals with annotation overlays"""
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        
        # Position signal (inverted for depth)
        axes[0].plot(-file_data['position'], label='Drilling Depth', color='blue')
        axes[0].set_ylabel('Depth (mm)')
        
        # Torque signal
        axes[1].plot(file_data['torque'], label='Torque Current', color='red')
        axes[1].set_ylabel('Torque (A)')
        
        # Thrust signal  
        axes[2].plot(file_data['thrust'], label='Thrust Current', color='green')
        axes[2].set_ylabel('Thrust (A)')
        
        # Step codes
        axes[3].step(range(len(file_data['steps'])), file_data['steps'], 
                    where='post', label='Step Codes', color='purple')
        axes[3].set_ylabel('Step')
        axes[3].set_xlabel('Sample Index')
        
    def enable_depth_annotation(self):
        """Enable click-and-drag annotation for depth points"""
        self.annotation_mode = True
        self.depth_markers = {'entry': None, 'transition': None, 'exit': None}
```

**Annotation Workflow Management**:
```python
class AnnotationWorkflowManager:
    """Manage annotation sessions and quality tracking"""
    
    def __init__(self):
        self.current_session = None
        self.quality_tracker = AnnotationQualityTracker()
        
    def start_annotation_session(self, file_list):
        """Initialize annotation session for file batch"""
        self.current_session = {
            'files': file_list,
            'current_index': 0,
            'annotations': {},
            'quality_scores': {},
            'session_start': datetime.now()
        }
    
    def save_annotation(self, file_id, depth_points, confidence_score):
        """Save expert annotation with quality metrics"""
        self.current_session['annotations'][file_id] = {
            'entry_depth': depth_points['entry'],
            'transition_depth': depth_points['transition'], 
            'exit_depth': depth_points['exit'],
            'annotator_confidence': confidence_score,
            'annotation_timestamp': datetime.now()
        }
```

## Command-Line Interface (CLI)

### Unified CLI Architecture

**Main CLI Entry Point**:
```bash
# Training platform CLI commands
uos-training-platform --help

# Data preparation workflow
uos-training-platform prepare --input-dir /data/setitec_files --validate --auto-label

# Model training workflow  
uos-training-platform train --config config.yaml --strategy patchtsmixer_24cv

# Model evaluation workflow
uos-training-platform evaluate --model-dir /models/trained --test-data /data/test

# Model conversion workflow
uos-training-platform convert --pytorch-model model.pth --output-format onnx

# Deployment workflow
uos-training-platform deploy --model model.onnx --target inference-system
```

**CLI Command Implementation**:
```python
import click
from training_platform import TrainingOrchestrator, DataValidator, ModelConverter

@click.group()
def cli():
    """UOS Drilling Training Platform CLI"""
    pass

@cli.command()
@click.option('--input-dir', required=True, help='Directory containing Setitec XLS files')
@click.option('--validate', is_flag=True, help='Run data quality validation')
@click.option('--auto-label', is_flag=True, help='Generate automatic labels')
def prepare(input_dir, validate, auto_label):
    """Prepare drilling data for training"""
    
    # Data validation
    if validate:
        validator = DataValidator()
        validation_results = validator.validate_directory(input_dir)
        click.echo(f"Validated {len(validation_results)} files")
    
    # Automatic labeling
    if auto_label:
        labeler = AutomaticLabeler()
        labels = labeler.process_directory(input_dir)
        click.echo(f"Generated labels for {len(labels)} files")

@cli.command()
@click.option('--config', required=True, help='Training configuration file')
@click.option('--strategy', default='patchtsmixer_24cv', help='Training strategy')
def train(config, strategy):
    """Train drilling depth estimation model"""
    
    orchestrator = TrainingOrchestrator(config, strategy)
    result = orchestrator.execute_full_pipeline()
    click.echo(f"Training completed. Model saved to: {result['model_path']}")
```

## Integration with Existing System

### Model Deployment Pipeline

**PyTorch → ONNX Conversion**:
```python
class ModelConversionPipeline:
    """Convert trained PyTorch models to ONNX for deployment"""
    
    def convert_patchtsmixer_ensemble(self, pytorch_models, output_dir):
        """Convert 24-fold CV ensemble to ONNX format"""
        
        onnx_models = []
        for i, model in enumerate(pytorch_models):
            # Convert individual fold model
            dummy_input = torch.randn(1, 17, 512)  # Batch, channels, sequence
            onnx_path = f"{output_dir}/fold_{i}.onnx"
            
            torch.onnx.export(
                model, dummy_input, onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['drilling_signals'],
                output_names=['depth_predictions']
            )
            onnx_models.append(onnx_path)
        
        # Create ensemble configuration
        ensemble_config = {
            'models': onnx_models,
            'voting_strategy': 'average',
            'confidence_weights': self._calculate_fold_weights(pytorch_models)
        }
        
        return ensemble_config
```

**Hot-Swappable Model Integration**:
```python
class ModelSwapManager:
    """Enable hot-swapping of models in production inference"""
    
    def __init__(self, inference_system):
        self.inference_system = inference_system
        self.model_registry = ModelRegistry()
        
    def deploy_new_model(self, model_package, rollback_enabled=True):
        """Deploy newly trained model with rollback capability"""
        
        # Backup current model
        if rollback_enabled:
            current_model = self.inference_system.get_current_model()
            self._backup_model(current_model)
        
        # Load and validate new model
        new_model = self._load_onnx_model(model_package['model_path'])
        validation_results = self._validate_model_performance(new_model)
        
        if validation_results['performance_ok']:
            # Hot-swap model in inference system
            self.inference_system.update_model(new_model)
            self._log_deployment(model_package, validation_results)
        else:
            raise ModelValidationError("New model failed performance validation")
```

## Implementation Timeline & Milestones

### Development Phases

**Phase 1: Data Pipeline Foundation (v1.3.x)** - 8-10 weeks
- Week 1-2: Setitec XLS ingestion and validation framework
- Week 3-4: Automated labeling system with confidence scoring
- Week 5-6: Data quality assurance and reporting system
- Week 7-8: Basic GUI annotation tool (file browser + signal plotting)
- Week 9-10: Integration testing and documentation

**Phase 2: Training Pipeline Implementation (v1.4.x)** - 6-8 weeks  
- Week 1-2: Abstract training strategy interface and PatchTSMixer implementation
- Week 3-4: Training orchestration engine and experiment tracking
- Week 5-6: CLI workflows and batch processing capabilities
- Week 7-8: PyTorch → ONNX conversion and validation pipeline

**Phase 3: Platform Integration (v1.5.x)** - 4-6 weeks
- Week 1-2: Model registry and version management (MLflow)
- Week 3-4: Hot-swappable deployment and production integration
- Week 5-6: Performance monitoring and A/B testing framework

### Success Criteria

**Technical Performance**:
- 95%+ data validation success rate for Setitec XLS files
- <5% annotation correction rate for high-confidence auto-labels
- <0.1% accuracy difference between PyTorch and ONNX models
- 90%+ successful training completion for new users

**User Experience**:
- 50%+ reduction in manual labeling time vs pure manual annotation
- New users productive within 2 weeks of training
- <30 minutes from trained model to production deployment

## Future Evolution & Extensibility

### Algorithm Extension Roadmap

**Phase 1 Extensions (v2.0.x)**:
- **Transformer Attention**: For long-sequence drilling operations
- **CNN-LSTM Hybrid**: For complex multi-modal drilling data
- **Ensemble Methods**: Combining multiple algorithm predictions

**Phase 2 Extensions (v2.1.x)**:
- **Neural ODE**: For continuous-time drilling modeling
- **Graph Neural Networks**: For multi-sensor fusion
- **Federated Learning**: For multi-site training without data sharing

### Platform Scalability

**Cloud Training Integration**:
- AWS SageMaker, GCP AI Platform, Azure ML support
- Auto-scaling training infrastructure
- Distributed training for large datasets (>1000 holes)

**Enterprise Features**:
- Multi-tenant architecture for multiple organizations
- Advanced analytics and drilling optimization recommendations
- REST/GraphQL APIs for third-party system integration
- Audit trails and compliance reporting

The UOS Drilling Training Platform represents a comprehensive solution that bridges the gap between domain expertise and machine learning capabilities, enabling industrial drilling professionals to leverage advanced ML techniques while maintaining focus on their core competencies in drilling operations.

---

# PART II: USER GUIDE FOR TRAINING PLATFORM

## User Guide Overview

This section provides complete end-user documentation for operating the UOS Drilling Training Platform, optimized for both direct consultation and AI assistant integration (Google Gemini, Claude, etc.).

## Quick Start Guide

### Getting Started with Training Platform

**System Access**:
```bash
# Web interface access
http://drilling-platform:5000/    # MLflow experiment tracking
http://drilling-platform:8501/    # Streamlit annotation GUI (if configured)

# Command-line access (advanced users)
ssh user@drilling-platform
cd /opt/uos-training-platform
```

**First-Time Setup**:
1. Organize your drilling files by material type
2. Validate data quality using automated checks
3. Generate automatic labels with hybrid labeling
4. Review and validate annotations in GUI
5. Start training with default configuration
6. Monitor training progress in MLflow
7. Evaluate and deploy trained models

### Common User Workflows

#### Basic Training Workflow (GUI-Focused)
1. **Upload Data** → GUI file browser
2. **Auto-Label** → Automatic step-code extraction
3. **Validate** → GUI annotation review
4. **Train** → One-click training start
5. **Deploy** → Export to production

#### Advanced Workflow (CLI-Focused)
1. **Batch Process** → CLI data validation
2. **Custom Config** → YAML parameter tuning
3. **Distributed Training** → Multi-GPU setup
4. **A/B Testing** → Model comparison
5. **API Integration** → Custom deployment

## Data Preparation Guide

### Understanding Setitec XLS Files

**Required Format**:
- Tab-separated text files with .xls extension
- Minimum 1000 data points per drilling operation
- Complete step sequences (1→2→N transitions)

**Essential Columns**:
- `Position (mm)`: Drilling depth progression
- `I Torque (A)`: Torque current measurements
- `I Thrust (A)`: Thrust current measurements
- `Step (nb)`: Drilling phase indicators

**Data Validation Commands**:
```bash
# Validate single file
uos-training-platform validate --file your_file.xls

# Validate entire directory
uos-training-platform validate --directory raw_drilling_files/

# Generate quality report
uos-training-platform quality-check --input-dir data/ --report
```

### Annotation Best Practices

**Understanding Drilling Signals**:
- **Position (Blue)**: Inverted depth progression
- **Torque (Red)**: Rotational force patterns
- **Thrust (Green)**: Downward force application
- **Step Codes (Purple)**: Phase transitions

**Depth Point Identification**:
- **Entry**: First material contact (Step 1→2)
- **Transition**: Material change or breakthrough (Step 2→N)
- **Exit**: Operation completion (final step)

**Quality Indicators**:
- High confidence (>80%): Auto-approve
- Medium confidence (50-80%): Review recommended
- Low confidence (<50%): Manual validation required

## Training Operations

### Starting Your First Training

**Using Default Configuration**:
```bash
# Basic training with defaults
uos-training-platform train --config default --data-dir your_data/

# Monitor progress
# Open: http://drilling-platform:5000/
```

**Understanding Training Metrics**:
- **Loss**: Should decrease over epochs
- **MAE**: Target <5mm for excellent performance
- **R²**: Target >0.90 for good correlation

### Advanced Training Options

**Custom Configuration Example**:
```yaml
# custom_training.yaml
model:
  strategy: "patchtsmixer_24cv"
  sequence_length: 512
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0005
  
validation:
  cv_folds: 24
  early_stopping: true
```

**Material-Specific Training**:
```bash
# Train on specific materials
uos-training-platform train --filter-materials concrete,steel

# Compare material performance
uos-training-platform evaluate --group-by material
```

## Model Deployment

### Export and Conversion

**PyTorch to ONNX**:
```bash
# Convert for production
uos-training-platform convert --model best_model --format onnx

# Validate conversion accuracy
uos-training-platform validate-conversion --pytorch-model original --onnx-model converted
```

### Production Integration

**MQTT Deployment**:
```bash
# Deploy to drilling system
uos-training-platform deploy --model production.onnx --target mqtt-system

# Enable monitoring
uos-training-platform monitor --enable --model production
```

## Troubleshooting Guide

### Common Issues and Solutions

**Data Format Errors**:
```
Error: Required column 'I Torque (A)' not found
Solution: Check exact column naming and capitalization
```

**Training Memory Issues**:
```
Error: CUDA out of memory
Solution: Reduce batch size or use gradient accumulation
```

**Poor Model Performance**:
```
Issue: R² < 0.80
Solutions:
- Increase training data
- Check data quality
- Adjust hyperparameters
```

### Performance Optimization Tips

1. **Data Quality**: Clean signals improve accuracy
2. **Balanced Dataset**: Equal material representation
3. **Sufficient Labels**: Minimum 100 examples per material
4. **Hyperparameter Tuning**: Use grid search for optimization

## Advanced Features

### Scaling to Large Datasets (1000+ holes)

**Memory-Efficient Processing**:
```bash
# Stream data during training
uos-training-platform train --streaming --buffer-size 1000

# Use data compression
uos-training-platform compress-data --format npz
```

### Cloud Training Setup

**AWS Configuration**:
```bash
# Configure and launch
uos-training-platform configure-cloud --provider aws
uos-training-platform cloud-train --instance-type p3.2xlarge
```

### Custom Model Architectures

**Available Strategies**:
- `patchtsmixer_24cv`: Default, proven accuracy
- `transformer_attention`: Complex patterns
- `ensemble_voting`: Maximum accuracy

## Reference Quick Links

### Command Reference
- `validate`: Data quality checks
- `auto-label`: Automatic annotation
- `train`: Model training
- `evaluate`: Performance assessment
- `convert`: Format conversion
- `deploy`: Production deployment
- `monitor`: Performance tracking

### Configuration Templates
Available in `/opt/uos-training-platform/configs/`:
- `default.yaml`: Standard training
- `advanced.yaml`: Custom parameters
- `cloud.yaml`: Distributed training
- `production.yaml`: Deployment settings

## AI Assistant Integration

### Optimized Query Patterns

**For Google Gemini or Claude**:
```
"I'm training on [material] with [N] files and getting [error/issue].
My configuration is [details]. What should I try?"
```

**Context Template**:
```
Context: UOS Drilling Training Platform v1.3.x
Data: [N] Setitec XLS files for [materials]
Hardware: [GPU/CPU specs]
Issue: [Specific problem]
```

This comprehensive guide enables both independent operation and AI-assisted troubleshooting for all platform features.