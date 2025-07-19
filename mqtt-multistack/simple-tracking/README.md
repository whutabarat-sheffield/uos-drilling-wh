# Simple MQTT Signal Tracking

Minimal signal tracking using existing tools with CSV files.

## What Changed

Instead of creating 3 new complex files (~800 lines), we:
- Added 15 lines to existing `uos_publish_json.py` 
- Created a 30-line signal monitor
- Use CSV files instead of a database
- Reuse existing Docker images

## Quick Start

1. **Prepare test data** in `data/` directory
2. **Copy config**: `cp ../../abyss/src/abyss/run/config/mqtt_conf_docker.yaml ./config/`
3. **Create tracking directory**: `mkdir -p tracking`
4. **Start the stack**: `docker-compose up -d`

## Monitor Progress

```bash
# Watch live statistics
docker-compose logs -f live-stats

# Check CSV files directly  
tail -f tracking/*.csv

# Get one-time report
docker-compose exec live-stats sh /app/compare_signals.sh
```

## How It Works

1. **Publisher**: Modified to inject `_signal_id` when `--track-signals` is used
2. **Monitor**: Simple Python script that extracts signal IDs and logs to CSV
3. **Analysis**: Shell script using `comm`, `join`, and `awk` for statistics

## CSV Format

**sent_signals.csv**:
```
signal_id,timestamp,toolbox_id,tool_id
```

**received_signals.csv**:
```
signal_id,timestamp,topic
```

## Portainer Deployment

Just paste the `docker-compose.yml` content into a new Portainer stack.

## Why This Is Better

- **15 lines** of changes vs 800 lines of new code
- **CSV files** are human-readable and grep-able
- **No database** complexity
- **Standard tools** (awk, join, comm) for analysis
- **Existing images** - no custom Dockerfile needed

Perfect for 10 signals/second testing!