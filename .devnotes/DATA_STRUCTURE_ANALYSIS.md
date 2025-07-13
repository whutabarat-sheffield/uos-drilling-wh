# Data Structure Analysis: Training Data Requirements

## Current Data Pipeline Analysis

### 1. **Primary Data Format: Setitec XLS Files**

**File Structure**: Tab-separated text files with `.xls` extension (not binary Excel)
- **Metadata Sections**: General parameters, cycle parameters, program parameters  
- **Data Section**: Identified by `"Position (mm)"` header row
- **Encoding**: Latin-1 for European character support

**Core Data Columns** (auto-detected from `dataparser.py`):
```python
# Primary features (always present)
'Position (mm)'        # Drill position (inverted to positive values)
'I Torque (A)'         # Primary torque measurement  
'I Thrust (A)'         # Primary thrust measurement
'I Torque Empty (A)'   # Empty/baseline torque correction
'Step (nb)'            # Program step identifier (critical for training)

# Additional features (firmware dependent)
'I Thrust Empty (A)'   # Empty/baseline thrust correction  
'Stop code'            # Operation stop codes
'Mem Torque min (A)'   # Memory torque minimum
'Mem Thrust min (A)'   # Memory thrust minimum
'Rotation Speed (rpm)' # Spindle speed (newer firmware)
'Feed Speed (mm/s)'    # Feed rate (newer firmware)
```

### 2. **Data Processing Pipeline**

**Current Inference Input Processing**:
```python
# From get_setitec_signals() in uos_depth_est_core.py
def get_setitec_signals(file_to_open: str) -> Tuple[np.ndarray, np.ndarray]:
    df = loadSetitecXls(file_to_open, version="auto_data")
    position = np.abs(df['Position (mm)'].values)        # Always invert to positive
    torque = df['I Torque (A)'].values
    torque_empty = df['I Torque Empty (A)'].values  
    torque_full = torque + torque_empty                   # Combined torque signal
    return position, torque_full

# Enhanced processing in process_for_analysis()
df['Position (mm)'] = df['Position (mm)'].abs()          # Position normalization
df['I Torque Total (A)'] = df['I Torque (A)'] + df['I Torque Empty (A)']
df['I Thrust Total (A)'] = df['I Thrust (A)'] + df['I Thrust Empty (A)']
```

### 3. **Step Code System (Critical for Training Labels)**

**Step Code Function**: Indicates drilling program phases
- **Step 1**: Entry phase (material contact, initial penetration)
- **Step 2**: Transition phase (material property changes) 
- **Step 3+**: Exit phase (breakthrough, completion)

**Step-based Label Extraction Functions**:
```python
# Key functions for extracting training labels from step codes
getStepCodeStartPos(fn, sc)    # Position where step code begins
getStepCodeFinalPos(fn, sc)    # Position where step code ends  
getStepCodeSampleCount(fn, sc) # Number of samples in step phase
```

**Critical Training Insight**: The current DEVNOTES mention system expects "2-stack, multi-step drilling configurations" - this indicates training labels are derived from step transitions.

### 4. **Current ML Model Input/Output**

**Model Architecture**: PatchTSMixer (24 cross-validation folds)
- **Input Shape**: `(batch_size, channels, sequence_length)` 
- **Channels**: 17 features (derived from position, torque, thrust, step data)
- **Sequence Length**: 512 samples (windowed time series)
- **Output**: Three-point depth estimation (entry, transition, exit positions)

**Inference Pipeline**:
```python
# From DepthInference class
class DepthInference:
    def infer3_common(self, data):
        # Three-point inference: entry, transition, exit
        enter_pos = inference_data_pipeline(df)     # Entry point detection
        trans_depth = add_cf2ti_point(df)          # Transition point detection  
        exit_depth = exit_estimation_pipeline(model, data)  # ML-based exit estimation
        return enter_pos, trans_depth, exit_depth
```

## Training Data Requirements Analysis

### 1. **Required Training Data Structure**

**File Organization**:
```
training_data/
├── raw/
│   ├── drilling_operation_001.xls    # Raw Setitec files
│   ├── drilling_operation_002.xls
│   └── ...
├── processed/
│   ├── features/                     # Extracted feature arrays
│   │   ├── position_data.npz
│   │   ├── torque_data.npz  
│   │   └── step_data.npz
│   └── labels/                       # Ground truth annotations
│       ├── entry_points.csv         # Entry depth labels
│       ├── transition_points.csv    # Transition depth labels
│       └── exit_points.csv          # Exit depth labels
└── splits/
    ├── train_files.txt              # Training file list
    ├── val_files.txt                # Validation file list  
    └── test_files.txt               # Test file list
```

### 2. **Label Generation Strategy**

**Option A: Step-Code Based Labeling (Automated)**
```python
# Automatic label extraction from step codes
def extract_training_labels(xls_file):
    labels = {}
    labels['entry_depth'] = getStepCodeStartPos(xls_file, sc=1)[1]      # Step 1 start
    labels['transition_depth'] = getStepCodeStartPos(xls_file, sc=2)[2] # Step 2 start  
    labels['exit_depth'] = getStepCodeFinalPos(xls_file, sc=2)[2]       # Step 2 end
    return labels
```

**Option B: Manual Annotation (Higher Quality)**
```csv
# labels/ground_truth.csv
filename,entry_depth_mm,transition_depth_mm,exit_depth_mm,material_type,drilling_config
drilling_001.xls,2.5,8.3,15.7,aluminum,2-stack
drilling_002.xls,1.8,7.9,14.2,titanium,2-stack
```

### 3. **Feature Engineering for Training**

**Input Feature Vector** (17 channels based on current system):
```python
def create_training_features(df):
    features = {
        'position': df['Position (mm)'].values,
        'torque_total': df['I Torque Total (A)'].values,
        'thrust_total': df['I Thrust Total (A)'].values,
        'torque_raw': df['I Torque (A)'].values,
        'thrust_raw': df['I Thrust (A)'].values,
        'step_code': df['Step (nb)'].values,
        # Derived features (computed from utils/functions/)
        'torque_gradient': np.gradient(torque_total),
        'thrust_gradient': np.gradient(thrust_total),
        'position_velocity': np.gradient(position),
        'torque_power': torque_total * rotation_speed,  # If available
        # Additional contextual features...
    }
    return np.stack(list(features.values()), axis=0)  # Shape: (17, n_samples)
```

### 4. **Data Quality Requirements**

**Minimum Dataset Requirements**:
- **File Count**: 100+ drilling operations per material/configuration type
- **Step Code Integrity**: All files must have valid step transitions (1→2→N)  
- **Signal Quality**: No missing values in critical channels (Position, Torque, Thrust)
- **Temporal Consistency**: Consistent sampling rates across files

**Data Validation Checks**:
```python
def validate_training_file(xls_path):
    df = loadSetitecXls(xls_path, "auto_data")
    
    # Required columns check
    required_cols = ['Position (mm)', 'I Torque (A)', 'I Thrust (A)', 'Step (nb)']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Step code validation  
    step_codes = df['Step (nb)'].unique()
    if len(step_codes) < 2:
        raise ValueError("Insufficient step codes for training")
        
    # Signal completeness
    if df[required_cols].isnull().any().any():
        raise ValueError("Missing values in critical signals")
        
    return True
```

## Key Clarifications Needed

### 1. **Ground Truth Labeling Method**
**Question**: How should training labels be determined?
- **Option A**: Automatic extraction from step codes (fast, potentially noisy)
- **Option B**: Manual annotation by drilling experts (slow, high quality)
- **Option C**: Hybrid approach with automatic pre-labeling + expert validation

### 2. **Model Training Strategy**  
**Question**: Should we maintain the 24-fold cross-validation approach?
- **Current System**: 24 CV folds (ensemble prediction)
- **Training Option**: Single best model + validation holdout
- **Ensemble Option**: Multiple models with different architectures

### 3. **Data Augmentation Requirements**
**Question**: What data augmentation techniques are appropriate?
- **Temporal**: Time warping, noise injection
- **Signal**: Amplitude scaling, baseline shift
- **Drilling Context**: Simulated tool wear, material variations

### 4. **Transfer Learning Strategy**
**Question**: How to leverage existing pre-trained models?
- **Fine-tuning**: Start from existing PatchTSMixer weights
- **Feature Extraction**: Use pre-trained features + new classifier  
- **Full Training**: Train from scratch with new data

## Recommended Training Data Organization

```
training/
├── data_preparation/
│   ├── validate_dataset.py          # Data quality validation
│   ├── extract_features.py          # Feature engineering pipeline
│   ├── generate_labels.py           # Label extraction (step-code or manual)
│   └── create_splits.py             # Train/val/test splitting
├── labeling_tools/
│   ├── manual_annotator.py          # GUI for manual depth annotation
│   ├── step_code_extractor.py       # Automatic step-based labeling
│   └── label_validator.py           # Cross-validation of labels
├── training_pipeline/
│   ├── pytorch_trainer.py           # PyTorch training implementation
│   ├── model_configs/               # Architecture configurations
│   └── experiment_tracking/         # MLflow/Weights&Biases integration
└── data/
    ├── raw_drilling_files/          # Original .xls files
    ├── processed_features/          # Preprocessed training arrays
    ├── annotations/                 # Ground truth labels
    └── splits/                      # Dataset splitting information
```

This analysis reveals that the training system should support both automatic step-code based labeling for rapid dataset creation, and manual annotation tools for higher-quality ground truth generation. The existing data pipeline provides a solid foundation that can be extended for training workflows.