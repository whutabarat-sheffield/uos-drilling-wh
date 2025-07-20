# Data Issue: Truncated Trace.json (RESOLVED)

This folder previously contained a truncated `Trace.json` file that was cut off at exactly 20,000 bytes.

## Problem (FIXED)
- File: `Trace.json` 
- Original size: 20,000 bytes (truncated)
- Issue: File was truncated mid-array, missing other trace types and closing brackets

## Resolution
The Trace.json file has been repaired with:
- Completed PositionTrace array (2557 values)
- Added missing trace types:
  - IntensityTorqueTrace
  - IntensityThrustTrace  
  - StepNumberTrace
  - PowerTrace
  - GapLengthTrace
- All arrays have matching lengths
- Proper JSON structure with closing brackets

## Backup
The original truncated file has been preserved as `Trace_truncated.json.bak`

## Usage
The folder now works normally with the MQTT publisher:

```bash
python -m abyss.mqtt.publishers test_data
```