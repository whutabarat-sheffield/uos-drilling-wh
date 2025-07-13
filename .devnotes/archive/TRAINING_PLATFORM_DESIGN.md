# Training Platform Design: Comprehensive Architecture

## Requirements Summary

**User Requirements:**
- **Hybrid Labeling**: Automated step-code extraction + expert validation GUI
- **Dataset Scale**: 300 labeled holes + 1000 unlabeled for pre-training (1300 total files)
- **Data Quality**: Automated validation with quality metrics and reporting
- **Training Approach**: Maintain 24-fold CV initially, design for future algorithm flexibility
- **User Interface**: GUI annotation tool + CLI batch processing workflows
- **User Expertise**: Basic ML knowledge (guided workflows, clear documentation)

## Architectural Philosophy: Future-Proof Flexibility

### 🎯 Core Design Principles

**1. Algorithm Agnostic Architecture**
```python
# Abstract training interface supporting multiple algorithms
class TrainingStrategy(ABC):
    @abstractmethod
    def prepare_data(self, dataset: Dataset) -> TrainingData: ...
    
    @abstractmethod  
    def train_model(self, data: TrainingData, config: TrainingConfig) -> TrainedModel: ...
    
    @abstractmethod
    def validate_model(self, model: TrainedModel, test_data: TestData) -> ValidationResults: ...

# Current implementation
class PatchTSMixerStrategy(TrainingStrategy):
    def train_model(self, data, config):
        # 24-fold cross-validation implementation
        
# Future implementations        
class TransformerStrategy(TrainingStrategy): ...
class CNNLSTMStrategy(TrainingStrategy): ...
class EnsembleStrategy(TrainingStrategy): ...
```

**2. Pluggable Model Registry**
```python
# Model registry for dynamic algorithm selection
class ModelRegistry:
    _strategies = {
        'patchtsmixer_24cv': PatchTSMixerStrategy,
        'patchtsmixer_single': SingleModelPatchTSMixerStrategy,
        'transformer_attention': TransformerStrategy,
        'ensemble_voting': EnsembleStrategy
    }
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> TrainingStrategy:
        return cls._strategies[strategy_name]()
        
    @classmethod  
    def register_strategy(cls, name: str, strategy_class: Type[TrainingStrategy]):
        cls._strategies[name] = strategy_class
```

**3. Configuration-Driven Training Pipeline**
```python
# Flexible configuration system
@dataclass
class TrainingConfiguration:
    # Algorithm selection
    model_strategy: str = 'patchtsmixer_24cv'
    
    # Data configuration  
    train_test_split: float = 0.8
    validation_split: float = 0.2
    sequence_length: int = 512
    feature_channels: int = 17
    
    # Training parameters (strategy-specific)
    training_params: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware configuration
    device: str = 'auto'  # 'cpu', 'cuda', 'mps', 'auto'
    batch_size: int = 32
    num_workers: int = 4
    
    # Experiment tracking
    experiment_name: str = 'depth_estimation_experiment'
    track_metrics: bool = True
    save_checkpoints: bool = True
```

## Complete Training Platform Architecture

### 🏗️ System Overview

```
training_platform/
├── data/                           # Data management layer
│   ├── ingestion/                  # File import and validation
│   ├── preprocessing/              # Feature extraction and cleaning  
│   ├── labeling/                   # Annotation and validation tools
│   ├── quality/                    # Automated quality assessment
│   └── storage/                    # Organized data storage
├── training/                       # Training orchestration layer
│   ├── strategies/                 # Algorithm implementations  
│   ├── pipelines/                  # Training workflow management
│   ├── experiments/                # Experiment tracking and versioning
│   └── validation/                 # Model performance evaluation
├── models/                         # Model management layer
│   ├── pytorch/                    # PyTorch training models
│   ├── onnx/                       # ONNX deployment models
│   ├── conversion/                 # PyTorch → ONNX pipeline
│   └── registry/                   # Model versioning and storage
├── interfaces/                     # User interface layer
│   ├── gui/                        # GUI annotation and monitoring tools
│   ├── cli/                        # Command-line workflows
│   ├── api/                        # REST API for integration
│   └── notebooks/                  # Jupyter tutorial notebooks
└── deployment/                     # Model deployment layer
    ├── validation/                 # Model performance testing
    ├── conversion/                 # Production format conversion
    ├── integration/                # Inference system integration
    └── monitoring/                 # Deployed model monitoring
```

### 📊 Data Management Architecture

**1. Data Ingestion & Validation Pipeline**
```python
class DataIngestionPipeline:
    def __init__(self):
        self.validators = [
            FileFormatValidator(),      # XLS format validation
            SignalQualityValidator(),   # Missing values, signal integrity
            StepCodeValidator(),        # Step transition validation  
            TemporalValidator(),        # Sampling rate consistency
            MetadataValidator()         # Required metadata presence
        ]
        
    def ingest_drilling_files(self, file_paths: List[str]) -> IngestionResults:
        """Process batch of drilling files with comprehensive validation"""
        results = IngestionResults()
        
        for file_path in file_paths:
            try:
                # Load and validate file
                raw_data = self.load_file(file_path)
                validation_report = self.validate_file(raw_data)
                
                if validation_report.is_valid:
                    processed_data = self.preprocess_file(raw_data)
                    results.add_success(file_path, processed_data, validation_report)
                else:
                    results.add_failure(file_path, validation_report)
                    
            except Exception as e:
                results.add_error(file_path, str(e))
                
        return results
```

**2. Automated Label Generation**
```python
class HybridLabelingPipeline:
    def __init__(self):
        self.auto_labeler = StepCodeLabelExtractor()
        self.confidence_estimator = LabelConfidenceEstimator()
        
    def generate_initial_labels(self, drilling_files: List[str]) -> LabelingResults:
        """Generate automatic labels with confidence scores"""
        results = LabelingResults()
        
        for file_path in drilling_files:
            # Extract step-code based labels
            auto_labels = self.auto_labeler.extract_labels(file_path)
            
            # Estimate confidence in automatic labeling
            confidence_score = self.confidence_estimator.score_labels(file_path, auto_labels)
            
            # Categorize for manual review based on confidence
            if confidence_score > 0.9:
                results.add_high_confidence(file_path, auto_labels, confidence_score)
            elif confidence_score > 0.7:
                results.add_medium_confidence(file_path, auto_labels, confidence_score)
            else:
                results.add_low_confidence(file_path, auto_labels, confidence_score)
                
        return results
        
class LabelConfidenceEstimator:
    def score_labels(self, file_path: str, labels: Dict) -> float:
        """Estimate confidence in automatic label extraction"""
        df = loadSetitecXls(file_path, "auto_data")
        
        confidence_factors = []
        
        # Step transition clarity
        step_transitions = self._analyze_step_transitions(df)
        confidence_factors.append(step_transitions)
        
        # Signal quality at transition points  
        signal_quality = self._analyze_signal_quality(df, labels)
        confidence_factors.append(signal_quality)
        
        # Consistency with expected drilling patterns
        pattern_consistency = self._analyze_drilling_patterns(df, labels)
        confidence_factors.append(pattern_consistency)
        
        return np.mean(confidence_factors)
```

### 🖥️ GUI Annotation Tool Architecture

**1. Main Annotation Interface**
```python
# Using PyQt6 for cross-platform GUI
class DrillAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.data_manager = AnnotationDataManager()
        self.visualization = DrillSignalVisualizer()
        
    def setup_ui(self):
        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QHBoxLayout(self.central_widget)
        
        # Left panel: File browser and annotation queue
        self.file_panel = AnnotationFilePanel()
        layout.addWidget(self.file_panel)
        
        # Center panel: Signal visualization and annotation
        self.annotation_panel = SignalAnnotationPanel()
        layout.addWidget(self.annotation_panel)
        
        # Right panel: Label validation and quality metrics
        self.validation_panel = LabelValidationPanel()
        layout.addWidget(self.validation_panel)
        
class SignalAnnotationPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_visualization()
        
    def setup_visualization(self):
        # Interactive matplotlib canvas for signal visualization
        self.figure = plt.Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Create subplots for different signals
        self.position_ax = self.figure.add_subplot(4, 1, 1)
        self.torque_ax = self.figure.add_subplot(4, 1, 2)  
        self.thrust_ax = self.figure.add_subplot(4, 1, 3)
        self.step_ax = self.figure.add_subplot(4, 1, 4)
        
        # Enable interactive annotation
        self.canvas.mpl_connect('button_press_event', self.on_click_annotation)
        self.canvas.mpl_connect('key_press_event', self.on_key_annotation)
        
    def load_drilling_file(self, file_path: str):
        """Load and visualize drilling data with automatic labels"""
        df = loadSetitecXls(file_path, "auto_data")
        auto_labels = self.extract_auto_labels(df)
        
        # Plot signals with automatic annotation markers
        self.plot_signals(df, auto_labels)
        
        # Enable interactive editing of label positions
        self.enable_label_editing(auto_labels)
```

**2. Batch Processing CLI Tools**
```python
# Command-line interface for batch operations
class TrainingCLI:
    def __init__(self):
        self.parser = self.create_parser()
        
    def create_parser(self):
        parser = argparse.ArgumentParser(description='UOS Drilling Training Platform')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Data preparation commands
        prep_parser = subparsers.add_parser('prepare', help='Data preparation workflows')
        prep_parser.add_argument('--input-dir', required=True, help='Directory with XLS files')
        prep_parser.add_argument('--output-dir', required=True, help='Output directory')
        prep_parser.add_argument('--validate', action='store_true', help='Run data validation')
        prep_parser.add_argument('--auto-label', action='store_true', help='Generate automatic labels')
        
        # Training commands
        train_parser = subparsers.add_parser('train', help='Model training workflows')
        train_parser.add_argument('--config', required=True, help='Training configuration file')
        train_parser.add_argument('--data-dir', required=True, help='Training data directory')
        train_parser.add_argument('--output-dir', required=True, help='Model output directory')
        train_parser.add_argument('--strategy', default='patchtsmixer_24cv', help='Training strategy')
        
        # Evaluation commands  
        eval_parser = subparsers.add_parser('evaluate', help='Model evaluation workflows')
        eval_parser.add_argument('--model-dir', required=True, help='Trained model directory')
        eval_parser.add_argument('--test-data', required=True, help='Test data directory')
        eval_parser.add_argument('--output-report', required=True, help='Evaluation report path')
        
        return parser

# Usage examples:
# python -m training_platform prepare --input-dir raw_data/ --output-dir processed/ --validate --auto-label
# python -m training_platform train --config configs/patchtsmixer_24cv.yaml --data-dir processed/ --output-dir models/
# python -m training_platform evaluate --model-dir models/experiment_001/ --test-data test_set/ --output-report results.json
```

### 🧠 Training Strategy Abstraction

**1. Flexible Training Configuration**
```yaml
# configs/patchtsmixer_24cv.yaml - Current approach
strategy: patchtsmixer_24cv
model_config:
  architecture: PatchTSMixer
  input_channels: 17
  sequence_length: 512
  patch_length: 16
  num_classes: 3  # entry, transition, exit

training_config:
  cross_validation:
    enabled: true
    n_folds: 24
    strategy: stratified
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping:
    patience: 10
    metric: val_loss

data_config:
  preprocessing:
    normalize_position: true
    combine_torque_signals: true
    sequence_windowing: true
    augmentation:
      noise_injection: 0.05
      time_warping: 0.1

# configs/transformer_single.yaml - Future approach example
strategy: transformer_single
model_config:
  architecture: Transformer
  input_channels: 17
  sequence_length: 512
  num_heads: 8
  num_layers: 6
  dropout: 0.1

training_config:
  cross_validation:
    enabled: false
  validation_split: 0.2
  batch_size: 64
  learning_rate: 0.0005
  epochs: 200
  optimizer: adamw
  scheduler: cosine_annealing
```

**2. Training Pipeline Orchestration**
```python
class TrainingOrchestrator:
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.strategy = ModelRegistry.get_strategy(config.model_strategy)
        self.experiment_tracker = ExperimentTracker(config.experiment_name)
        
    def run_training_pipeline(self, data_dir: str, output_dir: str) -> TrainingResults:
        """Execute complete training pipeline with experiment tracking"""
        
        # Phase 1: Data preparation
        self.log_phase("Data Preparation")
        dataset = self.load_and_validate_dataset(data_dir)
        training_data = self.strategy.prepare_data(dataset)
        
        # Phase 2: Model training
        self.log_phase("Model Training")
        trained_model = self.strategy.train_model(training_data, self.config)
        
        # Phase 3: Model validation
        self.log_phase("Model Validation") 
        validation_results = self.strategy.validate_model(trained_model, training_data.test_set)
        
        # Phase 4: Model conversion (PyTorch → ONNX)
        self.log_phase("Model Conversion")
        onnx_model = self.convert_to_onnx(trained_model)
        
        # Phase 5: Performance comparison
        self.log_phase("Performance Validation")
        conversion_validation = self.validate_onnx_conversion(trained_model, onnx_model)
        
        # Save results and artifacts
        results = TrainingResults(
            pytorch_model=trained_model,
            onnx_model=onnx_model,
            validation_metrics=validation_results,
            conversion_validation=conversion_validation,
            config=self.config
        )
        
        self.save_training_artifacts(results, output_dir)
        return results
```

### 🔄 Model Lifecycle Management

**1. Version Control and Experiment Tracking**
```python
class ModelVersionManager:
    def __init__(self, storage_backend='local'):  # Future: 's3', 'gcs', etc.
        self.storage = self._create_storage_backend(storage_backend)
        self.metadata_db = ModelMetadataDB()
        
    def register_model(self, model: TrainedModel, metadata: ModelMetadata) -> ModelVersion:
        """Register new model version with complete metadata"""
        
        # Generate version ID
        version_id = self._generate_version_id(metadata)
        
        # Store model artifacts
        pytorch_path = self.storage.store_pytorch_model(model.pytorch_model, version_id)
        onnx_path = self.storage.store_onnx_model(model.onnx_model, version_id)
        
        # Store training configuration and metrics
        config_path = self.storage.store_config(model.config, version_id)
        metrics_path = self.storage.store_metrics(model.validation_metrics, version_id)
        
        # Register in metadata database
        model_version = ModelVersion(
            version_id=version_id,
            pytorch_path=pytorch_path,
            onnx_path=onnx_path,
            config_path=config_path,
            metrics_path=metrics_path,
            metadata=metadata,
            created_at=datetime.now()
        )
        
        self.metadata_db.register_version(model_version)
        return model_version
        
    def compare_models(self, version_a: str, version_b: str) -> ModelComparison:
        """Compare performance metrics between model versions"""
        
        model_a = self.metadata_db.get_version(version_a)
        model_b = self.metadata_db.get_version(version_b) 
        
        # Load validation metrics
        metrics_a = self.storage.load_metrics(model_a.metrics_path)
        metrics_b = self.storage.load_metrics(model_b.metrics_path)
        
        # Generate comparison report
        return ModelComparison(
            version_a=model_a,
            version_b=model_b,
            metrics_comparison=self._compare_metrics(metrics_a, metrics_b),
            performance_diff=self._calculate_performance_diff(metrics_a, metrics_b)
        )
```

## Integration with Existing Roadmap

### 🚀 Version Planning Integration

**v1.3.x: Training Foundation** (8-10 weeks)
- **v1.3.0**: Data ingestion pipeline and validation framework
- **v1.3.1**: Hybrid labeling system with auto-generation
- **v1.3.2**: GUI annotation tool (basic version)

**v1.4.x: Training Pipeline** (6-8 weeks)  
- **v1.4.0**: PatchTSMixer training strategy implementation
- **v1.4.1**: CLI batch processing workflows
- **v1.4.2**: Model conversion and validation pipeline

**v1.5.x: Platform Integration** (4-6 weeks)
- **v1.5.0**: Integration with existing inference system
- **v1.5.1**: Model versioning and experiment tracking
- **v1.5.2**: Production deployment automation

**v2.0.x: Advanced Training Platform** (8-12 weeks)
- **v2.0.0**: Multi-strategy training support (transformer, ensemble, etc.)
- **v2.0.1**: Advanced GUI with batch annotation and quality analytics
- **v2.0.2**: Cloud training support and distributed computing

### 📊 Resource Requirements

**Development Effort:**
- **Data Pipeline**: 3-4 weeks (1 FTE)
- **GUI Development**: 4-6 weeks (1 FTE with UI experience)
- **Training Abstraction**: 3-4 weeks (1 FTE with ML expertise)
- **CLI Tools**: 2-3 weeks (1 FTE)
- **Integration & Testing**: 2-3 weeks (1 FTE)

**Infrastructure Requirements:**
- **Training Hardware**: GPU support for model training (RTX 3080+ or cloud equivalent)
- **Storage**: ~500GB for 1300 drilling files + training artifacts
- **Development Tools**: MLflow/Weights&Biases for experiment tracking

**User Training Materials:**
- **Tutorial Notebooks**: Step-by-step training workflows
- **Video Guides**: GUI annotation tool usage
- **Best Practices Guide**: Data organization and labeling strategies
- **Troubleshooting Guide**: Common issues and solutions

This architecture provides a complete, future-proof training platform that starts with the familiar 24-fold CV approach while enabling seamless migration to advanced training strategies as the project evolves. The modular design ensures that users can adopt new algorithms without disrupting their existing workflows or data preparation processes.