# Abyss

A Python package for drilling data analysis with real-time MQTT processing capabilities. The package includes both a core depth estimation library and a distributed MQTT-based processing system for Setitec electric drills.

## Contents
 - [Dependencies](#Dependencies)
 - [Installation](#Installation)
 - [Core Library Structure](#core-library-structure)
 - [MQTT Processing System](#mqtt-processing-system)
 - [Feature Checklist](#Checklist)
 - [Examples](#Examples)
 - [Depth Estimation](#Depth-Estimation)
 - [Notes](#Notes)

## Dependencies

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
- [h5py](https://docs.h5py.org/en/stable/)
- [nptdms](https://nptdms.readthedocs.io/en/stable/)
- [scipy](https://scipy.org/)
- [pywt](https://pywavelets.readthedocs.io/en/latest/)
- ~~[pwlf](https://pypi.org/project/pwlf/)~~ -> Newer depth estimation method implemented. No longer a core requirement
    + BreakpointFit class still left in for legacy support.
    + May be removed in the future
- [noisereduce](https://github.com/timsainb/noisereduce)
- [glob](https://docs.python.org/3/library/glob.html)
- [scaleogram](https://github.com/alsauve/scaleogram)
- [pyarrow](https://arrow.apache.org/docs/python/index.html)

## Installation

The package can be imported using pip. Change to the directory where the files have been downloaded and run the following command.

```
python setup.py install
```

## Core Library Structure
- [dataparser.py](src/dataparser.py)
    + Load and repackage data files
    + Support for TDMS, MAT and spreadsheets
- [plotting.py](src/plotting.py)
    + Quickly loading + plotting data files
    + Inspecting the contents of files
- [factory.py](src/factory.py)
    + Factors classes to attempt to load in whatever file is given using functions from dataparser.py
    + Attempts to infer file type and context from features in file structure
- [filters.py](src/filters.py)
    + Wrapper classes for filtering data
- [modelling.py](src/modelling.py)
    + Applying/fitting models to data
    + Wrappers for analysis functions
- [toolcodes.py](src/toolcodes.py)
    + Enum of tool codes from the Setitec documentation
    + NEEDS TO BE UPDATED

## MQTT Processing System

The MQTT processing system enables real-time analysis of drilling data streams at 100+ messages per second.

### Key Components

- **DrillingDataAnalyser**: Main orchestrator for MQTT message processing
- **ProcessingPool**: Parallel processing using ProcessPoolExecutor (10 workers)
- **MessageBuffer**: Thread-safe buffering with deduplication
- **SimpleMessageCorrelator**: Time-window based message correlation
- **ResultPublisher**: Configurable result publishing with validation
- **SimpleThroughputMonitor**: System health monitoring

### Quick Start

1. **Configure MQTT settings** in `mqtt_conf_local.yaml`:
```yaml
mqtt:
  broker:
    host: "localhost"
    port: 1883
  processing:
    workers: 10
    model_id: 4
  depth_validation:
    negative_depth_behavior: "warning"
```

2. **Run the MQTT processor**:
```bash
python src/abyss/run/mqtt_processor.py --config mqtt_conf_local.yaml
```

### Features

- **Parallel Processing**: 10-worker pool handles 0.5s depth inference
- **Configurable Validation**: Handle negative depths (publish/skip/warning)
- **Real-time Monitoring**: Track throughput and system health
- **Graceful Shutdown**: Clean worker termination and resource cleanup
- **Auto-reconnection**: Resilient MQTT connection handling

For detailed architecture documentation, see [docs/MQTT_ARCHITECTURE.md](docs/MQTT_ARCHITECTURE.md).

## Features to Add
### Data Parser
- [x] Support for Setitec spreadsheets
    + [x] Load Setitec spreadsheets into Pandas DataFrame
    + [ ] Support all recorded firmware versions (ONGOING AS MORE ARE FOUND)
    + [x] Support for files missing column headers and metadata
    + [ ] Support for files containing corrupted data
- [ ] Support for JSON files/format
- [x] Rolling Gradient method
    + [x] Implement and test method
    + [x] Filtering of data to help remove artifacts in the rolling gradient
    + [x] Add support for two element search window
    + [ ] Automatic selection of 2nd search window based on observed features of expected data
        * [x] 2nd peak where smallest gradient occurs
    + [ ] Correction due to rolling window

### Utilities
 - [x] Load several XLS files into a single pandas dataframe
 - [x] Find at what positions the step code changes.
 - [x] Find Program Values by Key
     + [x] Specific functions for DEP and AV
 - [x] Sort files by in-file datetime
 - [ ] Sort fles by firmware version

### Depth Estimation

Depth estimation is calculated based on the distance between two identified peaks in the [rolling window](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html) gradient of the torque signal. To reduce the number of spikes in the signal, the torque is smoothed using a [Wiener](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html) and a [Tukey](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html) window. The Tukey window is to remove aggressive spikes at the start and end of the torque. The Weiner filter removes some noise.

The rolling gradient is calculated using a sliding window in which the gradient of the contained data is calculated. The following plot is an example of the torque and the rolling gradient at different window sizes

![E00401009F45AF14_18080018_ST_2109_27-no-empty](https://user-images.githubusercontent.com/46482002/215822816-91d4de1d-8c6a-46e7-8c92-f9efdfa2f218.png)

Zooming in on the gradient the influence of window size is clearer. The window size is in number of samples. For smaller windows, small changes in the signal become more obvious and the estimated gradient is higher. Increasing the window size to 20 reduces its sensitivity to these small changes making it easier to identify the main changes in the signal. **As a rule of thumb, a window size of 20 to 30 is a good place to start with new data**

![E00401009F45AF14_18080018_ST_2109_27-gradient-zoom](https://user-images.githubusercontent.com/46482002/215828836-39df7494-a291-4f07-bebd-8de91d5eb687.png)

The depth estimation works by finding two peaks in the gradient and finding the distance between them. The first reference point is found towards the start of the signal and is ideally where the torque first enters the material. The user controls where it searches for this point using the *xstartA* parameter (standing for x-period for starting point A). **As a rule of them, setting this to 10.0 mm is a good start based on the experiments so far.** The first reference point is set as where the peak in the gradient is found. This is known as xA.

The user then provides an expected depth (*depth_exp*). This doesn't have to be accurate, just enough to get the second search window within the right area. This can be from the design documents or from experience. The centre of the second search window (right-hand red rectangle) is the *depth_exp* away from *xA* (the length of the black arrow). The parameter *depth_win* controls how far around this new centre point we search for the second reference point known as *xB*. Searching for the second reference point is the same as the first. The estimated depth is the distance between *xA* and *xB*.

![E00401009F45AF14_18080018_ST_2109_27-parameters](https://user-images.githubusercontent.com/46482002/215833587-d890a300-a039-4c18-94f8-8b36c7830378.png)

## Depth Correction
### Pull-back
An impact of using rolling gradient is the peaks in the gradient are always ahead of the causing changes in the data. The following plot demonstrates this. The blue lines are the torque signal after being smoothed at different window sizes. The red line is the gradient at different window sizes. The black vertical line is where the Setitec program changes from Step 0 to Step 1. This is a good reference point as it doesn't require detecting it from the signal. The green arrows are the distance from the nearest gradient peak to this step change. This distance is the amount the depth estimation needs to be corrected by. This is enabled by default in the depth_est_rolling function but can be explicitly enabled with *correct_dist* flag. To use it, the user needs to supply either the index where this step change occurs or the entire step code vector. **This method doesn't work if the program only has one step**.

![E00401009F45AF14_18080018_ST_2109_27-depth-correct-sc](https://user-images.githubusercontent.com/46482002/215836590-f454e27c-02a8-4e1a-a319-b73d87c96668.png)

### RPCA (Robust Principal Component Analysis)

RPCA takes a bank of signals and finds what's common between them and what's different. For this purpose, it is used as a denoising method where what's common is kept and the differences are discarded. The class [R_pca](https://github.com/D-B-Miller/ACSE-EADU_data_analytics/blob/aadfa3f71ecebd845ba41ff50acaaf0f805dc3dc/scripts/abyss/src/modelling.py#L251) takes a bank of column wise signals and produces a Low rank (what's common) and a Scatter (what's different matrix). The Low rank matrix is the set of signals we'd then use to calculate depth.

The bank of signals are torque signals over a tool's lifetime clipped to a common length. This is required to stack them into a matrix.

## Examples
### Sorting Setitec XLS files
Setitec filenames are composed of 3 main parts

head type_family name_head count
 
The head type is a unique alpha-numeric sequence related to the type of tool head that's being used. This tool head is unique in the sense that no two tool heads are the same. The family name is often several two or three character alpha-numeric sequences delimited by dashes. The important bit used for sorting files is the headcount. This is a typically two part incrementing counter showing how many times the given tool head has been used. The parts represent the global and local head counter and are both also stored in the file metadata. The counters first increments the global head counter and then the local head counter.
 
e.g.
x_x_1_1
x_x_1_2
x_x_1_3
x_x_1_4
...
x_x_2_1
x_x_2_2
x_x_2_3
x_x_2_4
 
There are also cases where it has 3 parts e.g x_x_2_1_1 which refers to a hold that was initially drilled into air and was then re-drilled. As the majority are just two parts, the focus is on them.

To sort the files you first sort the global head counter and then the local head counter. If it's just a single global head counter value then just sort the local value. The head counters can be found by parsing and splitting the filename into parts using the [os](https://docs.python.org/3/library/os.html) package.

```python
import os
local_hole = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
global_hole = int(os.path.splitext(os.path.basename(path))[0].split('_')[-2])
```
To check for 3 part head counters, you can do something like this.
 
```python
is_three = all([c.isnumeric() for c in os.path.splitext(os.path.basename(path))[0].split('_')[-3:]])
```
### Load Setitec XLS file
There are two main ways of loading the data from XLS files. The default approach is "auto" where it tries to parse all the data in the file into sections and returns them as a list. The list is a mix of dictionaries for the metadata and one pandas DataFrame containing the run data at the end of the file. The metadata sections can be merged together incorrectly as different firmware versions can be structured differently.

```python
from abyss import dataparser as dp
import numpy as np
data = dp.loadSetitecXls(path,version="auto")
```

Which is why the second way to just load the run data is added. This approach, specified by setting version to "auto_data", searches for where 'Position (mm)' occurs in the file and reads the data into a Pandas DataFrame. This skips ALL the metadata in the file and just gets you the data.

```python
from abyss import dataparser as dp
import numpy as np
data = dp.loadSetitecXls(path,version="auto_data")
```

### Depth Estimation

```python
from abyss import dataparser as dp
import numpy as np

# load the setitec XLS file
# it returns the header information as dictionaries and data as pandas DataFrames
# the last element is the data in the process data
data  = dp.loadSetitecXLS(path)[-1]

# get the position data
pos = data['Position (mm)'].values.flatten()
# absolute the position data otherwise the algorithm has issues finding the gradient
pos = np.abs(pos)

# load the torque data
torque = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()

# perform the dpeth estimation from the rolling gradient
# NA : Window size for smoothing filter. If NB isn't given, then NA is used for the rolling gradient window size
# depth_exp : Expected depth thickness
# depth_win : Search window
# default : Flag to get the function to default to upper limits when it fails to find peaks.
dest = depth_est_rolling(torque,pos,NA=30,depth_exp=31.0,depth_win=4.5,default=True)
```

#### Segmented depth estimation
```python
from abyss.rolling_gradient import depth_est_segment
import numpy as np

# load the setitec XLS file
# it returns the header information as dictionaries and data as pandas DataFrames
# the last element is the data in the process data
data  = dp.loadSetitecXLS(path)[-1]

# get the position data
pos = data['Position (mm)'].values.flatten()
# absolute the position data otherwise the algorithm has issues finding the gradient
pos = np.abs(pos)

# load the torque data
torque = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()

# perform the dpeth estimation from the rolling gradient
# NA : Window size for smoothing filter. If NB isn't given, then NA is used for the rolling gradient window size
# depth_exp : Expected depth thickness
# depth_win : Search window
# default : Flag to get the function to default to upper limits when it fails to find peaks.
dest = depth_est_rolling(torque,pos,NA=30,depth_exp=31.0,depth_win=4.5,default=True)

### Energy Estimation
Calculating the energy estimation for a single file

```python
from abyss import dataparser as dp
import numpy as np
import energy_estimation as em

# load only the run data
data  = dp.loadSetitecXLS(path,version="auto_data")
e = em.get_energy_estimation(data['I Torque (A)'].values.flatten(),data['I Torque Empty (A)'].values.flatten(),data['I Thrust (A)'].values.flatten())
```

For multiple files

```python
from abyss import dataparser as dp
import numpy as np
import energy_estimation as em
from glob import glob

# function to calculate energy for a single file
def est_energy(path):
    data  = dp.loadSetitecXLS(path,version="auto_data")
    return em.get_energy_estimation(data['I Torque (A)'].values.flatten(),data['I Torque Empty (A)'].values.flatten(),data['I Thrust (A)'].values.flatten())

# apply est_energy to each found file
# here files are sorted according to final number of filename which is related to local head count
# also map can be replaced by a for loop
elist = map(est_energy, sorted(glob('*.xls'),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])))
```

### Notes
#### TDMS file

TDMS files are a standard produced by National Instrument. The package [nptdms](https://nptdms.readthedocs.io/en/stable/) is a cross platform package that handles loading in the data and making it accessible as a HDF5 file or a Pandas.DataFrame. The TDMS functions in dataparser are mainly wrappers to simplify accessing other stuff.

When loaded, the file presents in a similar structure to a HDF5 file.
 - groups -> keys
 - properties -> attrs

When accessed as a HDF5 file, a HDF5 file is created in the local directory. The datasets are copied across under the default setting.

When converting to a Pandas.DataFrame, there is an option called time_index. When True, the index of the frame is converted to floating point timestamps. This process can take a while and doubles the size of the returned array compared to using integer index. Also converting to timestamps introduces a high number of NaN values. This is the cause of the doubling in size and also has duplicate timestamps. Not sure why this occurs but the NaN values can be ignored using the dropna method with the returned DataFrame.
