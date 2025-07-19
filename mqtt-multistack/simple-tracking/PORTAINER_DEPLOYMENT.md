# Portainer Deployment Guide

This guide explains how to deploy the signal tracking system in Portainer.

## Prerequisites

1. `uos-publish-json:latest` image must be available in your Docker registry
2. `mqtt-broker_toolbox-network` external network must exist
3. `mqtt-broker` container must be running

## Deployment Steps

### Option 1: Using Pre-built Image with Modified Publisher

If you have built a custom image that includes the modified `uos_publish_json.py` with signal tracking:

1. Deploy the stack using `docker-compose.portainer.yml`
2. The system will start tracking signals automatically

### Option 2: Manual Setup with Volume Initialization

1. **Initialize volumes** (run this once):
   ```bash
   docker-compose -f docker-compose.portainer-init.yml up
   docker-compose -f docker-compose.portainer-init.yml down
   ```

2. **Copy your test data** to the `test-data` volume:
   ```bash
   # Find the volume name
   docker volume ls | grep test-data
   
   # Copy data files
   docker run --rm -v <volume-name>:/data -v ./data:/source busybox cp -r /source/* /data/
   ```

3. **Deploy in Portainer**:
   - Create a new stack in Portainer
   - Copy the contents of `docker-compose.portainer.yml`
   - Deploy the stack

### Option 3: Build Custom Image (Recommended)

1. **Create a Dockerfile** for the modified publisher:
   ```dockerfile
   FROM uos-publish-json:latest
   COPY ./abyss/src/abyss/run/uos_publish_json.py /app/uos_publish_json.py
   ```

2. **Build and push the image**:
   ```bash
   docker build -t uos-publish-json-tracked:latest .
   docker push uos-publish-json-tracked:latest
   ```

3. **Update `docker-compose.portainer.yml`** to use the new image:
   ```yaml
   publisher:
     image: uos-publish-json-tracked:latest
   ```

4. **Deploy in Portainer**

## Volume Structure

The stack uses named volumes for portability:

- `tracking-data`: Shared volume for CSV files
- `test-data`: Test data files
- `config-data`: MQTT configuration
- `scripts-data`: Python scripts

## Monitoring

Once deployed, you can:

1. **View live statistics**:
   ```bash
   docker logs -f live-stats
   ```

2. **Check signal monitor logs**:
   ```bash
   docker logs signal-monitor
   ```

3. **Access tracking data**:
   ```bash
   # List tracking files
   docker run --rm -v tracking-data:/tracking busybox ls -la /tracking/
   
   # View sent signals
   docker run --rm -v tracking-data:/tracking busybox head /tracking/sent_signals.csv
   ```

## Troubleshooting

### "Mounts denied" Error
This error occurs when using relative paths. Always use named volumes in Portainer.

### No Signals Received
1. Check if the publisher is running: `docker ps | grep publisher`
2. Verify MQTT broker connectivity: `docker logs signal-monitor`
3. Ensure the modified `uos_publish_json.py` is being used

### Missing Test Data
The test data needs to be manually copied to the `test-data` volume or included in a custom image.