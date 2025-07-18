# Exact Timestamp Matching Fix

## Problem Description

The original MQTT publisher (`uos_publish_json.py`) was modifying the SourceTimestamp values in messages before publishing them. This caused the exact matching system to fail because:

1. The publisher replaced original timestamps with the current time
2. Messages were published with random delays between them
3. Even messages from the same source file would have different timestamps

This resulted in the listener reporting only "Incomplete message group" errors and no successful matches.

## Solution

We created a simplified publisher (`uos_publish_json_simple.py`) that:
- Preserves original SourceTimestamp values without modification
- Publishes all three message types (Result, Trace, Heads) immediately without delays
- Validates that all three files have matching timestamps before publishing

## Usage

### Running the Simplified Publisher

```bash
# Basic usage - preserves original timestamps
python src/abyss/run/uos_publish_json_simple.py /path/to/data/folder

# With debug output to see timestamps being published
python src/abyss/run/uos_publish_json_simple.py /path/to/data/folder --debug-timestamps

# Legacy mode (updates timestamps like the original publisher)
python src/abyss/run/uos_publish_json_simple.py /path/to/data/folder --update-timestamps

# Custom configuration
python src/abyss/run/uos_publish_json_simple.py /path/to/data/folder -c config/mqtt_conf_local.yaml
```

### Testing Exact Matching

Use the test script to verify that exact matching is working:

```bash
# Run the test listener for 30 seconds
python src/abyss/run/test_exact_matching.py

# Run for a different duration
python src/abyss/run/test_exact_matching.py -d 60

# With debug logging
python src/abyss/run/test_exact_matching.py --log-level DEBUG
```

The test script will show:
- Each message received with its SourceTimestamp
- Whether complete matches are found (all three message types with same timestamp)
- A summary of complete vs incomplete matches

## Example Output

### Simplified Publisher Output
```
[14:36:56]: Published data on OPCPUBSUB/ILLL502033771/setitec001/ResultManagement
[14:36:56]: Published data on OPCPUBSUB/ILLL502033771/setitec001/ResultManagement/Trace
[14:36:56]: Published data on OPCPUBSUB/ILLL502033771/setitec001/AssetManagement/Heads
Publishing dataset with SourceTimestamp: 2024-01-18T10:30:45Z
```

### Test Script Output
```
[0.123s] Received result from ILLL502033771/setitec001 with timestamp: 2024-01-18T10:30:45Z
[0.125s] Received trace from ILLL502033771/setitec001 with timestamp: 2024-01-18T10:30:45Z
[0.127s] Received heads from ILLL502033771/setitec001 with timestamp: 2024-01-18T10:30:45Z
✓ COMPLETE MATCH for ILLL502033771/setitec001 with timestamp 2024-01-18T10:30:45Z
```

## Key Differences from Original Publisher

| Feature | Original (`uos_publish_json.py`) | Simplified (`uos_publish_json_simple.py`) |
|---------|----------------------------------|-------------------------------------------|
| Timestamp handling | Replaces with current time | Preserves original |
| Message order | Random shuffle | Fixed order (Result→Trace→Heads) |
| Delays | Random delays between each message | No delays between message types |
| Timestamp validation | Checks within each file | Validates across all three files |

## Troubleshooting

### JSON Decode Errors
If you see errors like:
```
JSON decode error: Expecting ',' delimiter: line 1 column 20001 (char 20000)
```

This indicates that some JSON files are truncated at exactly 20,000 characters. Check the source data files for corruption.

### Timestamp Mismatches
If the publisher reports:
```
Timestamp mismatch in /path/to/data:
  result: 2024-01-18T10:30:45Z
  trace: 2024-01-18T10:30:46Z
  heads: 2024-01-18T10:30:45Z
```

This means the source files don't have matching timestamps. The exact matching system requires all three files to have identical SourceTimestamp values.

## Recommendations

1. **For Production**: Use the simplified publisher without the `--update-timestamps` flag
2. **For Testing**: Run the test script alongside the publisher to verify matching
3. **Data Validation**: Ensure source data files have consistent timestamps across Result, Trace, and Heads
4. **Performance**: The simplified publisher sends messages faster (no delays), which may increase broker load