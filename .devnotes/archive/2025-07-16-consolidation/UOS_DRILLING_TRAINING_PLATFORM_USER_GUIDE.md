# UOS Drilling Training Platform: Complete User Guide for AI Assistant Integration

## CONTEXT FOR AI ASSISTANT

**System**: UOS Drilling Depth Estimation Training Platform v1.3.x+
**Technology Stack**: MLflow + DVC + PyTorch/ONNX dual architecture
**Data Requirements**: Setitec XLS drilling files (300 labeled + 1000 unlabeled holes)
**User Profile**: Industrial drilling professionals with basic ML knowledge
**Deployment**: Portainer multi-stack Docker environment
**AI Integration**: Optimized for Google Gemini interrogation and user support

## QUICK REFERENCE: Common User Questions → Section Mapping

**Data & File Management**:
- "How do I prepare my drilling data?" → Section 2.1-2.3
- "What file format does the system expect?" → Section 2.1
- "System says my data quality is poor, how do I fix it?" → Section 7.1

**Annotation & Labeling**:
- "How do I annotate drilling data?" → Section 3.1-3.3
- "How do I validate my annotations are correct?" → Section 3.3
- "Can the system automatically label my data?" → Section 2.2

**Training & Models**:
- "How do I start training a new model?" → Section 4.1
- "My training failed with memory error, what should I do?" → Section 7.2
- "How do I compare two different models I've trained?" → Section 5.2

**Deployment & Production**:
- "How do I export my trained model for production use?" → Section 6.1
- "How do I integrate my model with the drilling system?" → Section 6.2
- "How do I monitor my model's performance?" → Section 6.3

**Scaling & Advanced**:
- "How do I scale up to train on 1000+ drilling holes?" → Section 8.2
- "Can I use different training algorithms?" → Section 8.1
- "How do I set up cloud training?" → Section 8.3

## 1. GETTING STARTED

### 1.1 System Overview for End Users

The UOS Drilling Training Platform enables you to create custom depth estimation models using your own drilling data. The platform combines automated data processing with expert validation tools, allowing you to train models that understand your specific drilling conditions, materials, and equipment.

**What You Can Accomplish**:
- Train custom models using your drilling operations data
- Improve accuracy for your specific materials and drilling conditions
- Deploy trained models to existing drilling estimation systems
- Continuously improve models with new drilling data

**Platform Components**:
- **Data Management**: Handles your Setitec XLS drilling files with automated validation
- **Annotation Tools**: GUI and CLI tools for labeling drilling phases (entry, transition, exit)
- **Training Pipeline**: Automated model training with experiment tracking
- **Model Deployment**: Seamless integration with existing drilling systems

### 1.2 User Access and System Requirements

**System Access**:
```bash
# Web interface access
http://drilling-platform:5000/    # MLflow experiment tracking
http://drilling-platform:8501/    # Streamlit annotation GUI

# Command-line access (advanced users)
ssh user@drilling-platform
cd /opt/uos-training-platform
```

**Required Skills**:
- **Basic**: Understanding of drilling operations and phases
- **Intermediate**: Familiarity with file management and basic computer operations
- **Advanced**: Basic understanding of machine learning concepts (optional)

**Hardware Requirements**:
- **Training Workstation**: GPU recommended (RTX 3070+ or equivalent)
- **Storage**: 500GB available space for datasets and models
- **Network**: Stable internet connection for cloud features (optional)

### 1.3 Data Organization and File Structure

**Required Directory Structure**:
```
project_data/
├── raw_drilling_files/           # Your original Setitec XLS files
│   ├── material_A/              # Organize by material type
│   ├── material_B/
│   └── material_C/
├── validated_data/              # System-processed files
├── annotations/                 # Expert labels and validations
├── trained_models/              # Your custom trained models
└── experiments/                 # Training experiment results
```

**File Naming Conventions**:
```
# Recommended naming for Setitec XLS files
{material}_{configuration}_{date}_{unique_id}.xls

Examples:
concrete_2stack_20240301_001.xls
steel_3stack_20240301_002.xls
masonry_2stack_20240302_001.xls
```

## 2. DATA PREPARATION

### 2.1 Setitec XLS File Requirements

**Expected File Format**: Setitec XLS files (tab-separated text with .xls extension)

**Required Data Columns**:
```
Position (mm)      # Drilling depth progression
I Torque (A)       # Torque current measurements  
I Thrust (A)       # Thrust current measurements
Step (nb)          # Drilling step codes (1, 2, 3, etc.)
```

**Additional Helpful Columns** (optional but recommended):
```
Torque Empty (A)   # Baseline torque measurements
Thrust Empty (A)   # Baseline thrust measurements
Time (s)           # Timestamp information
Speed (rpm)        # Drilling speed (if available)
```

**Data Quality Requirements**:
- **Minimum Length**: 1000 data points per drilling operation
- **Complete Sequences**: Valid step transitions (1→2→N)
- **No Missing Values**: Critical columns must be complete
- **Consistent Sampling**: Regular time intervals between measurements

**Example Data Validation**:
```bash
# Check your file format
uos-training-platform validate --file your_drilling_file.xls

# Validate entire directory
uos-training-platform validate --directory raw_drilling_files/

# Generate quality report
uos-training-platform validate --directory raw_drilling_files/ --report
```

### 2.2 Automated Label Generation (Hybrid Labeling)

**How Automatic Labeling Works**:
The system analyzes your step codes to identify drilling phases automatically:

1. **Entry Point**: Transition from Step 1 to Step 2
2. **Transition Point**: Transition from Step 2 to final step (Step 3, 4, etc.)
3. **Exit Point**: Completion of final drilling step

**Label Confidence Levels**:
- **High Confidence (>80%)**: Clear step transitions, good signal quality → Auto-approve
- **Medium Confidence (50-80%)**: Acceptable transitions, minor issues → Review recommended
- **Low Confidence (<50%)**: Unclear transitions, poor signal → Manual validation required

**Running Automatic Labeling**:
```bash
# Generate labels for all files
uos-training-platform auto-label --input-dir raw_drilling_files/ --output-dir annotations/

# Generate labels with confidence filtering
uos-training-platform auto-label --input-dir raw_drilling_files/ --min-confidence 0.6

# Review auto-generated labels
uos-training-platform review-labels --annotations-dir annotations/
```

**Understanding Confidence Scores**:
- **Signal Quality**: Clear, stable measurements increase confidence
- **Step Transitions**: Sharp, well-defined transitions increase confidence  
- **Data Completeness**: No missing values or gaps increase confidence
- **Historical Patterns**: Consistency with similar drilling operations

### 2.3 Data Quality Validation and Troubleshooting

**Automated Quality Checks**:
```bash
# Run comprehensive quality assessment
uos-training-platform quality-check --input-dir raw_drilling_files/

# Check specific quality aspects
uos-training-platform quality-check --check-format --check-completeness --check-signals

# Generate detailed quality report
uos-training-platform quality-check --input-dir raw_drilling_files/ --detailed-report quality_report.html
```

**Common Data Issues and Solutions**:

**Issue 1: Missing Data Columns**
```
Error: Required column 'I Torque (A)' not found
Solution: Check column names in XLS file, ensure exact spelling and capitalization
```

**Issue 2: Invalid Step Sequences**
```
Warning: Step sequence 1→3 detected, missing Step 2
Solution: Check drilling operation, may indicate equipment malfunction or data collection error
```

**Issue 3: Insufficient Data Length**
```
Warning: File contains only 500 data points, minimum 1000 required
Solution: Combine multiple short operations or exclude from training dataset
```

**Issue 4: High Signal Noise**
```
Warning: Signal-to-noise ratio below threshold
Solution: Check drilling equipment calibration, consider signal filtering options
```

## 3. ANNOTATION INTERFACE

### 3.1 GUI Annotation Tool (Primary Interface)

**Starting the Annotation Tool**:
```bash
# Launch GUI annotation interface
uos-training-platform annotate-gui

# Launch with specific project directory
uos-training-platform annotate-gui --project-dir /path/to/your/project
```

**Interface Layout**:
```
┌─────────────────┬───────────────────────────────┬─────────────────┐
│                 │                               │                 │
│   File Browser  │     Signal Visualization      │  Validation     │
│   & Queue       │     & Annotation Area         │  & Metrics      │
│                 │                               │                 │
│   - File list   │   - Position signal           │ - Confidence    │
│   - Progress    │   - Torque signal             │ - Quality score │
│   - Filters     │   - Thrust signal             │ - Validation    │
│                 │   - Step codes                │ - Comments      │
└─────────────────┴───────────────────────────────┴─────────────────┘
```

**Basic Annotation Workflow**:
1. **Select File**: Choose drilling file from the file browser
2. **Review Auto-Labels**: System shows automatically generated depth points
3. **Validate Signals**: Examine position, torque, thrust, and step signals
4. **Adjust if Needed**: Drag depth markers to correct positions
5. **Confirm Quality**: Review confidence scores and signal quality
6. **Save Annotation**: Save validated labels for training

### 3.2 Reading and Understanding Drilling Signals

**Signal Interpretation Guide**:

**Position Signal (Blue Line)**:
- **Interpretation**: Drilling depth progression (typically inverted, negative values)
- **Key Features**: Smooth progression during drilling, plateaus during non-drilling phases
- **What to Look For**: Consistent downward progression, absence of sudden jumps

**Torque Signal (Red Line)**:
- **Interpretation**: Rotational force during drilling
- **Key Features**: Increases during material engagement, stable during steady drilling
- **What to Look For**: Sharp increases at material transitions, consistent levels during uniform drilling

**Thrust Signal (Green Line)**:
- **Interpretation**: Downward force applied during drilling
- **Key Features**: Controlled application, varies with material resistance
- **What to Look For**: Coordinated changes with torque, appropriate levels for material type

**Step Codes (Purple Steps)**:
- **Interpretation**: Drilling phase indicators (1=approach, 2=drilling, 3+=completion)
- **Key Features**: Discrete transitions between phases
- **What to Look For**: Logical progression (1→2→3), clear transition points

### 3.3 Depth Point Annotation and Validation

**Understanding Depth Points**:

**Entry Depth**:
- **Definition**: Point where drill bit first contacts material
- **Signal Characteristics**: Sudden increase in torque and thrust, step transition 1→2
- **Annotation**: Click and drag red marker to precise contact point

**Transition Depth**:
- **Definition**: Point where drilling changes character (material change, break-through)
- **Signal Characteristics**: Change in torque/thrust patterns, step transition 2→3
- **Annotation**: Click and drag yellow marker to transition point

**Exit Depth**:
- **Definition**: Point where drilling operation completes
- **Signal Characteristics**: Decrease in torque/thrust, final step transition
- **Annotation**: Click and drag green marker to completion point

**Validation Checklist**:
- ✅ **Signal Consistency**: Do depth points align with signal changes?
- ✅ **Step Code Alignment**: Do depth points match step transitions?
- ✅ **Physical Plausibility**: Do depths make sense for drilling operation?
- ✅ **Quality Metrics**: Are confidence scores acceptable (>0.6)?

**Common Annotation Errors**:
- **Early Entry**: Marking entry before actual material contact
- **Late Exit**: Marking exit after drilling has stopped
- **Missing Transitions**: Not identifying material changes or break-through points
- **Signal Misinterpretation**: Confusing noise spikes with actual drilling events

## 4. TRAINING WORKFLOWS

### 4.1 Basic Training (Guided Workflow)

**Step-by-Step Training Process**:

**Step 1: Prepare Training Dataset**
```bash
# Verify data quality and completeness
uos-training-platform prepare-training --input-dir raw_drilling_files/ --validate

# Create training dataset with auto-labels
uos-training-platform prepare-training --input-dir raw_drilling_files/ --auto-label --min-confidence 0.7
```

**Step 2: Configure Training Parameters**
```bash
# Use default configuration (recommended for beginners)
uos-training-platform train --config default --data-dir training_data/

# Customize basic parameters
uos-training-platform train --epochs 50 --batch-size 16 --learning-rate 0.001
```

**Step 3: Monitor Training Progress**
```bash
# View training in web interface
# Open: http://drilling-platform:5000/
# Navigate to: Experiments > Current Training Run

# Monitor training metrics:
# - Training Loss (should decrease over time)
# - Validation Accuracy (should increase over time)  
# - Epoch Progress (current training progress)
```

**Step 4: Evaluate Training Results**
```bash
# Generate training report
uos-training-platform evaluate --model latest --test-data test_dataset/

# Compare with baseline model
uos-training-platform compare --model1 baseline --model2 latest
```

**Expected Training Timeline**:
- **Small Dataset (100-300 files)**: 2-4 hours on GPU, 8-12 hours on CPU
- **Medium Dataset (300-700 files)**: 4-8 hours on GPU, 12-24 hours on CPU
- **Large Dataset (700-1300 files)**: 8-16 hours on GPU, 24-48 hours on CPU

### 4.2 Advanced Training (Custom Parameters)

**Custom Training Configuration**:
```yaml
# training_config.yaml
model:
  strategy: "patchtsmixer_24cv"
  sequence_length: 512
  feature_channels: 17
  
data:
  train_split: 0.7
  validation_split: 0.2
  test_split: 0.1
  augmentation: true
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0005
  optimizer: "adamw"
  scheduler: "cosine"
  
validation:
  cv_folds: 24
  metrics: ["mae", "rmse", "r2"]
  early_stopping: true
  patience: 10
```

**Advanced Training Commands**:
```bash
# Train with custom configuration
uos-training-platform train --config training_config.yaml

# Train with specific material focus
uos-training-platform train --filter-materials concrete,steel --config custom.yaml

# Resume interrupted training
uos-training-platform train --resume --experiment-id exp_12345

# Hyperparameter optimization
uos-training-platform optimize --config base_config.yaml --trials 20
```

**Advanced Feature Options**:
- **Data Augmentation**: Synthetic variation generation for robust training
- **Transfer Learning**: Initialize with pre-trained models for faster convergence
- **Ensemble Training**: Multiple model strategies for improved accuracy
- **Distributed Training**: Multi-GPU training for large datasets

### 4.3 Cross-Validation and Model Selection

**Understanding 24-Fold Cross-Validation**:
The system uses 24-fold cross-validation to ensure robust model performance:

1. **Data Splitting**: Dataset divided into 24 equal portions
2. **Training Cycles**: 24 models trained, each using 23 folds for training, 1 for validation
3. **Ensemble Creation**: Final model combines predictions from all 24 models
4. **Performance Assessment**: Average performance across all folds provides reliable estimate

**Interpreting Cross-Validation Results**:
```bash
# View detailed CV results
uos-training-platform cv-results --experiment-id exp_12345

# Typical output:
# Fold 1: MAE=2.3mm, RMSE=3.1mm, R²=0.91
# Fold 2: MAE=2.1mm, RMSE=2.9mm, R²=0.93
# ...
# Average: MAE=2.2mm, RMSE=3.0mm, R²=0.92
# Std Dev: MAE=0.3mm, RMSE=0.4mm, R²=0.02
```

**Model Selection Criteria**:
- **Average Performance**: Lower MAE and RMSE, higher R² indicate better accuracy
- **Consistency**: Lower standard deviation indicates more reliable performance
- **Fold Stability**: Similar performance across folds suggests robust model
- **Validation Trend**: Improving validation metrics indicate good learning

## 5. MODEL EVALUATION

### 5.1 Performance Metrics and Interpretation

**Primary Performance Metrics**:

**Mean Absolute Error (MAE)**:
- **Definition**: Average absolute difference between predicted and actual depths
- **Interpretation**: Lower values better, measured in mm
- **Acceptable Range**: <5mm excellent, 5-10mm good, >10mm needs improvement

**Root Mean Square Error (RMSE)**:
- **Definition**: Square root of average squared errors
- **Interpretation**: Penalizes larger errors more heavily than MAE
- **Acceptable Range**: <7mm excellent, 7-15mm good, >15mm needs improvement

**R-Squared (R²)**:
- **Definition**: Proportion of variance explained by the model
- **Interpretation**: Closer to 1.0 is better (1.0 = perfect prediction)
- **Acceptable Range**: >0.90 excellent, 0.80-0.90 good, <0.80 needs improvement

**Material-Specific Performance**:
```bash
# Evaluate performance by material type
uos-training-platform evaluate --group-by material

# Example output:
# Concrete: MAE=1.8mm, RMSE=2.4mm, R²=0.94
# Steel: MAE=2.8mm, RMSE=3.5mm, R²=0.89
# Masonry: MAE=3.2mm, RMSE=4.1mm, R²=0.86
```

### 5.2 Model Comparison and Selection

**Comparing Multiple Models**:
```bash
# Compare two trained models
uos-training-platform compare --model1 baseline_v1 --model2 custom_v2

# Compare multiple models
uos-training-platform compare --models baseline_v1,custom_v2,ensemble_v3

# Generate comparison report
uos-training-platform compare --models all --output comparison_report.html
```

**Model Comparison Criteria**:

**Accuracy Comparison**:
- **Overall Performance**: Compare MAE, RMSE, R² across all test data
- **Material-Specific**: Identify which model performs best for specific materials
- **Edge Cases**: Evaluate performance on difficult or unusual drilling conditions

**Robustness Assessment**:
- **Cross-Validation Stability**: Compare standard deviations across CV folds
- **Outlier Handling**: Evaluate performance on unusual or extreme measurements
- **Noise Tolerance**: Test performance with varying signal quality

**Practical Considerations**:
- **Inference Speed**: Model prediction time (important for real-time applications)
- **Model Size**: Storage and memory requirements
- **Deployment Complexity**: Ease of integration with existing systems

### 5.3 Validation Procedures and Quality Assurance

**Model Validation Workflow**:

**Step 1: Hold-Out Test Set Validation**
```bash
# Reserve 20% of data for final testing (never used in training)
uos-training-platform split-data --test-split 0.2 --stratify-by material

# Evaluate model on held-out test set
uos-training-platform validate --model trained_model --test-set holdout_test/
```

**Step 2: Real-World Validation**
```bash
# Test on completely new drilling operations
uos-training-platform validate --model trained_model --new-data recent_operations/

# Compare predictions with actual measurements
uos-training-platform validate --model trained_model --ground-truth actual_depths.csv
```

**Step 3: Production Validation**
```bash
# Deploy model in shadow mode (predictions logged but not used)
uos-training-platform deploy --model trained_model --mode shadow

# Monitor production performance
uos-training-platform monitor --model deployed_model --duration 30days
```

**Validation Quality Checklist**:
- ✅ **Test Set Independence**: Test data never seen during training
- ✅ **Representative Sampling**: Test set covers all materials and conditions
- ✅ **Statistical Significance**: Sufficient test samples for reliable estimates
- ✅ **Real-World Correlation**: Lab/simulation results match field performance
- ✅ **Edge Case Coverage**: Performance validated on unusual conditions

## 6. DEPLOYMENT

### 6.1 Model Export and Format Conversion

**PyTorch to ONNX Conversion**:
```bash
# Convert trained PyTorch model to ONNX format
uos-training-platform convert --model trained_model --format onnx --output production_model.onnx

# Validate conversion accuracy
uos-training-platform validate-conversion --pytorch-model trained_model --onnx-model production_model.onnx

# Optimize ONNX model for production
uos-training-platform optimize --onnx-model production_model.onnx --target cpu
```

**Model Export Formats**:
- **ONNX**: Primary format for production deployment (recommended)
- **TorchScript**: Alternative PyTorch deployment format
- **TensorFlow Lite**: For mobile/edge deployment (future support)
- **Custom Format**: Specialized formats for specific deployment targets

**Export Quality Verification**:
```bash
# Compare PyTorch vs ONNX predictions
uos-training-platform verify-export --pytorch-model trained.pth --onnx-model exported.onnx

# Expected output:
# Prediction Accuracy: 99.98% match
# Maximum Difference: 0.001mm
# Average Difference: 0.0003mm
# Status: PASS - Safe for production deployment
```

### 6.2 Production Integration with Drilling System

**Integration Workflow**:

**Step 1: Model Packaging**
```bash
# Create deployment package
uos-training-platform package --model production_model.onnx --config deployment_config.yaml

# Deployment package contents:
# - production_model.onnx (trained model)
# - model_metadata.json (version, training info)
# - inference_config.yaml (runtime configuration)
# - validation_report.pdf (quality assurance)
```

**Step 2: System Integration**
```bash
# Deploy to existing drilling inference system
uos-training-platform deploy --package model_package.zip --target drilling-system

# Hot-swap deployment (zero downtime)
uos-training-platform deploy --package model_package.zip --mode hot-swap --fallback current_model
```

**Step 3: Production Validation**
```bash
# Start with shadow mode (log predictions, don't use)
uos-training-platform production-test --mode shadow --duration 24hours

# Graduate to A/B testing (50/50 split)
uos-training-platform production-test --mode ab-test --duration 1week

# Full deployment
uos-training-platform deploy --mode production --model validated_model
```

**MQTT Integration Configuration**:
```yaml
# mqtt_integration.yaml
mqtt:
  broker: "drilling-broker:1883"
  input_topic: "drilling/data/signals"
  output_topic: "drilling/predictions/depth"
  
model:
  path: "production_model.onnx"
  batch_size: 1
  timeout_ms: 100
  
monitoring:
  performance_topic: "drilling/monitoring/model"
  log_predictions: true
  alert_threshold_ms: 500
```

### 6.3 Performance Monitoring and Model Maintenance

**Production Monitoring Setup**:
```bash
# Enable production monitoring
uos-training-platform monitor --enable --model production_model

# Configure monitoring alerts
uos-training-platform monitor --configure --alerts performance,accuracy,drift

# View monitoring dashboard
# Open: http://drilling-platform:5000/monitoring
```

**Key Monitoring Metrics**:

**Performance Metrics**:
- **Inference Latency**: Time to generate predictions (target: <100ms)
- **Throughput**: Predictions per second (target: >10 pred/sec)
- **Error Rate**: Failed predictions (target: <0.1%)
- **Resource Usage**: CPU/memory consumption

**Accuracy Monitoring**:
- **Prediction Drift**: Changes in prediction patterns over time
- **Confidence Scores**: Model uncertainty levels
- **Outlier Detection**: Unusual predictions requiring review
- **Ground Truth Comparison**: When actual depths become available

**Model Maintenance Schedule**:

**Daily Monitoring**:
- Check inference latency and error rates
- Review unusual predictions or confidence scores
- Monitor system resource usage

**Weekly Analysis**:
- Analyze prediction drift trends
- Review model performance metrics
- Collect feedback from drilling operators

**Monthly Evaluation**:
- Compare predictions with actual drilling results
- Assess need for model retraining
- Plan data collection for next training cycle

**Model Refresh Triggers**:
- **Performance Degradation**: Accuracy drops below acceptable thresholds
- **New Materials**: Introduction of new drilling materials or conditions
- **Equipment Changes**: Modifications to drilling equipment or procedures
- **Seasonal Patterns**: Regular retraining to capture temporal variations

## 7. TROUBLESHOOTING DATABASE

### 7.1 Data Issues and Solutions

**Problem: File Format Not Recognized**
```
Error: Unable to parse XLS file - invalid format
```
**Diagnosis**: 
- Check if file is actually tab-separated text (not Excel binary format)
- Verify file extension is .xls
- Confirm file is not corrupted

**Solutions**:
```bash
# Check file format
file your_drilling_file.xls

# Convert Excel to tab-separated format
uos-training-platform convert-format --input excel_file.xlsx --output converted_file.xls

# Validate file structure
uos-training-platform validate --file converted_file.xls --verbose
```

**Problem: Missing Required Columns**
```
Error: Required column 'I Torque (A)' not found in file
```
**Diagnosis**:
- Column names don't match expected format
- Extra spaces or special characters in column headers
- Different language or measurement units

**Solutions**:
```bash
# List all columns in file
uos-training-platform inspect --file your_file.xls --columns

# Map custom column names
uos-training-platform configure --column-mapping "Torque Current:I Torque (A)"

# Use flexible column detection
uos-training-platform prepare --flexible-columns --input your_file.xls
```

**Problem: Data Quality Issues**
```
Warning: High noise level detected in signals
Warning: Missing values in critical columns
```
**Diagnosis**:
- Equipment calibration issues
- Data collection interruptions
- Signal processing problems

**Solutions**:
```bash
# Apply signal filtering
uos-training-platform preprocess --filter-noise --interpolate-missing

# Exclude problematic sections
uos-training-platform preprocess --exclude-sections "1000:1500,2200:2400"

# Generate quality report
uos-training-platform quality-report --input-dir data/ --output quality_analysis.html
```

### 7.2 Training Failures and Recovery

**Problem: Out of Memory During Training**
```
Error: CUDA out of memory. Tried to allocate 2.94 GiB
```
**Diagnosis**:
- Batch size too large for available GPU memory
- Model too complex for hardware
- Memory leak in training process

**Solutions**:
```bash
# Reduce batch size
uos-training-platform train --batch-size 8 --config your_config.yaml

# Use gradient accumulation
uos-training-platform train --batch-size 4 --accumulation-steps 4

# Train on CPU (slower but more memory)
uos-training-platform train --device cpu --batch-size 16
```

**Problem: Training Not Converging**
```
Warning: Loss not decreasing after 20 epochs
Warning: Validation accuracy plateaued
```
**Diagnosis**:
- Learning rate too high or too low
- Insufficient training data
- Model architecture mismatch
- Data quality issues

**Solutions**:
```bash
# Adjust learning rate
uos-training-platform train --learning-rate 0.0001 --scheduler cosine

# Increase training data
uos-training-platform augment-data --input training_data/ --augment-factor 2

# Try different model architecture
uos-training-platform train --strategy transformer_attention --epochs 50
```

**Problem: Training Interrupted**
```
Error: Training stopped due to system shutdown
```
**Solutions**:
```bash
# Resume from last checkpoint
uos-training-platform resume --experiment-id exp_12345

# Restart with checkpoint frequency
uos-training-platform train --save-frequency 10 --config your_config.yaml

# Monitor training remotely
uos-training-platform train --remote-monitoring --email your@email.com
```

### 7.3 Deployment and Integration Issues

**Problem: ONNX Conversion Accuracy Loss**
```
Warning: ONNX model predictions differ from PyTorch by >0.1mm
```
**Diagnosis**:
- Unsupported operations in ONNX
- Numerical precision differences
- Model architecture incompatibility

**Solutions**:
```bash
# Use conservative conversion settings
uos-training-platform convert --conservative --precision float32

# Validate conversion step-by-step
uos-training-platform debug-conversion --model your_model.pth

# Use TorchScript as alternative
uos-training-platform convert --format torchscript
```

**Problem: High Inference Latency**
```
Warning: Model inference taking >500ms per prediction
```
**Diagnosis**:
- Model too complex for production requirements
- Suboptimal hardware configuration
- Inefficient model implementation

**Solutions**:
```bash
# Optimize model for inference
uos-training-platform optimize --model production_model.onnx --target-latency 100ms

# Use model quantization
uos-training-platform quantize --model production_model.onnx --precision int8

# Profile performance bottlenecks
uos-training-platform profile --model production_model.onnx --report performance.html
```

**Problem: MQTT Integration Failure**
```
Error: Failed to connect to MQTT broker
Error: Message format not recognized
```
**Diagnosis**:
- Network connectivity issues
- MQTT broker configuration problems
- Message format incompatibility

**Solutions**:
```bash
# Test MQTT connectivity
uos-training-platform mqtt-test --broker drilling-broker:1883

# Validate message format
uos-training-platform mqtt-validate --topic drilling/data/signals

# Update integration configuration
uos-training-platform configure-mqtt --broker-config updated_mqtt.yaml
```

## 8. ADVANCED TOPICS

### 8.1 Custom Model Architectures and Algorithms

**Available Training Strategies**:

**PatchTSMixer (Default)**:
- **Use Case**: Standard drilling operations with regular patterns
- **Strengths**: Fast training, reliable performance, proven accuracy
- **Configuration**: 24-fold CV, 512 sequence length, 17 feature channels

**Transformer Attention**:
- **Use Case**: Complex drilling patterns with long-range dependencies
- **Strengths**: Captures complex temporal relationships
- **Configuration**: Multi-head attention, positional encoding

**CNN-LSTM Hybrid**:
- **Use Case**: Multi-modal data with spatial and temporal patterns
- **Strengths**: Combines feature extraction with sequence modeling
- **Configuration**: Convolutional layers + LSTM layers

**Ensemble Methods**:
- **Use Case**: Maximum accuracy for critical applications
- **Strengths**: Combines multiple model predictions
- **Configuration**: Voting, stacking, or boosting strategies

**Switching Training Strategies**:
```bash
# Train with Transformer architecture
uos-training-platform train --strategy transformer_attention --config transformer_config.yaml

# Train ensemble model
uos-training-platform train --strategy ensemble_voting --models patchtsmixer,transformer,cnn_lstm

# Compare strategies on your data
uos-training-platform compare-strategies --data your_training_data/ --strategies all
```

### 8.2 Large Dataset Management (1000+ Holes)

**Scaling Strategies for Large Datasets**:

**Data Organization**:
```bash
# Organize data by material and time
uos-training-platform organize --input large_dataset/ --by material,date --validate

# Create balanced training splits
uos-training-platform split --input organized_data/ --balance-by material --train 0.7 --val 0.2 --test 0.1
```

**Distributed Training**:
```bash
# Multi-GPU training
uos-training-platform train --strategy patchtsmixer_24cv --gpus 4 --distributed

# Cloud training on multiple nodes
uos-training-platform cloud-train --nodes 2 --gpus-per-node 2 --config large_dataset_config.yaml
```

**Memory-Efficient Processing**:
```bash
# Stream data during training (don't load all at once)
uos-training-platform train --streaming --buffer-size 1000 --config memory_efficient.yaml

# Use data compression
uos-training-platform compress-data --input large_dataset/ --output compressed_data/ --format npz
```

**Incremental Training**:
```bash
# Train on subset, then expand
uos-training-platform train --initial-subset 300 --expand-by 200 --max-data 1000

# Continual learning (add new data to existing model)
uos-training-platform continue-training --base-model existing_model --new-data recent_operations/
```

### 8.3 Cloud Training and Distributed Computing

**Cloud Training Setup**:

**AWS Configuration**:
```bash
# Configure AWS credentials
uos-training-platform configure-cloud --provider aws --region us-east-1

# Launch cloud training
uos-training-platform cloud-train --instance-type p3.2xlarge --data s3://your-bucket/drilling-data/
```

**Google Cloud Configuration**:
```bash
# Configure GCP credentials
uos-training-platform configure-cloud --provider gcp --region us-central1

# Launch cloud training with TPUs
uos-training-platform cloud-train --tpu-type v3-8 --data gs://your-bucket/drilling-data/
```

**Azure Configuration**:
```bash
# Configure Azure credentials
uos-training-platform configure-cloud --provider azure --region eastus

# Launch cloud training
uos-training-platform cloud-train --vm-size Standard_NC6s_v3 --data azure://container/drilling-data/
```

**Cost Optimization**:
```bash
# Use spot instances for cost savings
uos-training-platform cloud-train --spot-instances --max-price 0.50

# Schedule training during off-peak hours
uos-training-platform schedule-training --start-time "02:00" --timezone UTC

# Auto-shutdown when training completes
uos-training-platform cloud-train --auto-shutdown --max-runtime 8hours
```

## 9. REFERENCE MATERIALS

### 9.1 Complete Command Reference

**Data Management Commands**:
```bash
# Validation and quality checks
uos-training-platform validate --file <file> [--verbose]
uos-training-platform validate --directory <dir> [--report]
uos-training-platform quality-check --input-dir <dir> [--detailed-report <file>]

# Data preparation
uos-training-platform prepare-training --input-dir <dir> [--validate] [--auto-label]
uos-training-platform auto-label --input-dir <dir> [--min-confidence <float>]
uos-training-platform organize --input <dir> --by <criteria> [--validate]

# Format conversion
uos-training-platform convert-format --input <file> --output <file>
uos-training-platform preprocess --filter-noise --interpolate-missing
```

**Training Commands**:
```bash
# Basic training
uos-training-platform train --config <config> --data-dir <dir>
uos-training-platform train --epochs <int> --batch-size <int> --learning-rate <float>

# Advanced training
uos-training-platform train --strategy <name> --config <config>
uos-training-platform resume --experiment-id <id>
uos-training-platform optimize --config <config> --trials <int>

# Evaluation
uos-training-platform evaluate --model <model> --test-data <dir>
uos-training-platform compare --models <model1>,<model2>,...
uos-training-platform cv-results --experiment-id <id>
```

**Deployment Commands**:
```bash
# Model conversion
uos-training-platform convert --model <model> --format <format> --output <file>
uos-training-platform optimize --onnx-model <model> --target <target>
uos-training-platform validate-conversion --pytorch-model <model> --onnx-model <model>

# Deployment
uos-training-platform package --model <model> --config <config>
uos-training-platform deploy --package <package> --target <target>
uos-training-platform deploy --mode <mode> --model <model>

# Monitoring
uos-training-platform monitor --enable --model <model>
uos-training-platform monitor --configure --alerts <types>
```

### 9.2 Configuration File Templates

**Basic Training Configuration**:
```yaml
# basic_training.yaml
model:
  strategy: "patchtsmixer_24cv"
  sequence_length: 512
  feature_channels: 17

data:
  input_dir: "training_data/"
  train_split: 0.7
  validation_split: 0.2
  test_split: 0.1

training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  optimizer: "adam"

validation:
  cv_folds: 24
  metrics: ["mae", "rmse", "r2"]
  early_stopping: true
  patience: 10
```

**Advanced Training Configuration**:
```yaml
# advanced_training.yaml
model:
  strategy: "transformer_attention"
  sequence_length: 1024
  feature_channels: 17
  attention_heads: 8
  hidden_size: 256

data:
  input_dir: "large_training_data/"
  augmentation:
    enabled: true
    noise_level: 0.05
    time_warping: true
  preprocessing:
    normalization: "z_score"
    filtering: "low_pass"

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0005
  optimizer: "adamw"
  scheduler: "cosine"
  gradient_clipping: 1.0

distributed:
  enabled: true
  gpus: 4
  strategy: "ddp"

logging:
  experiment_name: "transformer_large_dataset"
  log_frequency: 10
  save_frequency: 20
```

**Production Deployment Configuration**:
```yaml
# production_deployment.yaml
model:
  path: "production_model.onnx"
  optimization: "cpu"
  precision: "float32"

inference:
  batch_size: 1
  timeout_ms: 100
  max_queue_size: 1000

mqtt:
  broker: "drilling-broker:1883"
  input_topic: "drilling/data/signals"
  output_topic: "drilling/predictions/depth"
  qos: 1

monitoring:
  enabled: true
  metrics: ["latency", "accuracy", "throughput"]
  alert_thresholds:
    latency_ms: 200
    error_rate: 0.01
  logging:
    level: "INFO"
    file: "/var/log/drilling-model.log"
```

### 9.3 Integration API Documentation

**Python API for Custom Integration**:
```python
from uos_training_platform import TrainingPlatform, ModelManager

# Initialize platform
platform = TrainingPlatform(config_path="config.yaml")

# Load and validate data
data = platform.load_data("drilling_files/")
validation_results = platform.validate_data(data)

# Train model
training_config = {
    "strategy": "patchtsmixer_24cv",
    "epochs": 50,
    "batch_size": 16
}
model = platform.train_model(data, training_config)

# Evaluate and deploy
metrics = platform.evaluate_model(model, test_data)
production_model = platform.convert_to_onnx(model)
platform.deploy_model(production_model, "production")
```

**REST API Endpoints**:
```bash
# Model management
GET /api/models                     # List all models
POST /api/models/train              # Start training
GET /api/models/{id}/status         # Training status
POST /api/models/{id}/deploy        # Deploy model

# Data management  
POST /api/data/upload               # Upload drilling files
GET /api/data/validate              # Validate dataset
POST /api/data/auto-label           # Generate labels

# Inference
POST /api/predict                   # Single prediction
POST /api/predict/batch             # Batch predictions
GET /api/monitor/metrics            # Performance metrics
```

## 10. GOOGLE GEMINI OPTIMIZATION GUIDE

### 10.1 Effective Query Patterns for AI Assistance

**Problem-Solution Query Pattern**:
```
"I'm having trouble with [specific issue]. My setup is [brief context]. 
What should I do?"

Example:
"I'm having trouble with training convergence. My setup is 500 concrete drilling 
files with PatchTSMixer strategy. The loss plateaued after 20 epochs. What should I do?"
```

**Step-by-Step Process Queries**:
```
"How do I [accomplish specific task] step by step?"

Example:
"How do I deploy my trained model to the production drilling system step by step?"
```

**Troubleshooting Queries**:
```
"I'm getting [exact error message]. What does this mean and how do I fix it?"

Example:
"I'm getting 'CUDA out of memory. Tried to allocate 2.94 GiB'. 
What does this mean and how do I fix it?"
```

### 10.2 Context Injection Templates for Drilling-Specific Queries

**Complete Context Template**:
```
Context: I'm working with the UOS Drilling Training Platform v1.3.x. 
I have [number] drilling files in Setitec XLS format for [materials]. 
I'm using [training strategy] with [hardware setup].

Question: [Your specific question]

Current status: [What you've tried or where you're stuck]
```

**Material-Specific Context**:
```
Drilling Context: Working with [material type] drilling operations, 
[stack configuration], typical depth range [X-Y mm]. 
Data quality is [assessment]. Using [equipment type].

Question: [Your specific question about this material]
```

**Error Context Template**:
```
Error Context: Command '[exact command]' failed with error '[exact error message]'. 
System: [OS and hardware]. Configuration: [relevant config details].
Previous steps: [what led to this error].

Question: How do I resolve this error?
```

### 10.3 Multi-Step Problem Solving with AI

**Complex Problem Breakdown Strategy**:

**Step 1: Problem Definition**
```
"I need to improve my model accuracy from 85% to 90% R² score. 
Current setup: 400 drilling files, mixed materials, PatchTSMixer strategy. 
What are the main approaches I should consider?"
```

**Step 2: Option Analysis**
```
"You suggested [data augmentation/architecture changes/hyperparameter tuning]. 
Which approach would be most effective for my specific case with mixed materials?"
```

**Step 3: Implementation Planning**
```
"I want to try [chosen approach]. What are the specific steps and configuration 
changes I need to make? What should I monitor during implementation?"
```

**Step 4: Validation and Iteration**
```
"I implemented [approach] and got [results]. The improvement was [amount]. 
Should I continue with this approach or try something different?"
```

**Iterative Improvement Pattern**:
1. **Baseline Assessment**: "What's my current performance and main limitations?"
2. **Prioritized Improvements**: "What's the highest-impact improvement I can make?"
3. **Implementation Guidance**: "How exactly do I implement this improvement?"
4. **Results Evaluation**: "How do I know if the improvement worked?"
5. **Next Steps**: "What should I optimize next?"

**Emergency Problem Solving**:
```
"URGENT: My production model stopped working with error [X]. 
I need to restore service quickly. What's the fastest recovery approach?"

Follow-up: "Now that service is restored, how do I prevent this from happening again?"
```

This comprehensive user guide enables both direct human consultation and AI-assisted support, providing multiple pathways to solve drilling training platform challenges efficiently and effectively.