# Multi-Stack Portainer Deployment Guide

## Overview

This guide explains how to deploy the UOS Drilling Depth Estimation system using separate Portainer stacks for each component, enabling proper cross-stack communication.

## Architecture

The system consists of three separate Portainer stacks:

1. **mqtt-broker** - MQTT message broker
2. **uos-depthest-listener** - Main processing application
3. **uos-publish-json** - Test data publisher

## Prerequisites

### 1. Create External Network

Before deploying any stacks, create a shared external network in Portainer:

**Via Portainer UI:**
1. Go to **Networks** → **Add Network**
2. Name: `toolbox-network`
3. Driver: `bridge`
4. Click **Create Network**

**Via Command Line:**
```bash
docker network create toolbox-network
```

### 2. Verify Network Creation

```bash
docker network ls | grep toolbox-network
```

## Stack Deployment Order

Deploy stacks in this order to ensure proper dependencies:

1. **mqtt-broker** (first)
2. **uos-depthest-listener** (second)
3. **uos-publish-json** (third)

## Stack Configurations

### Stack 1: mqtt-broker

**Stack Name:** `mqtt-broker`

```yaml
version: '3.8'

services:
  mqtt-broker:
    image: eclipse-mosquitto:latest
    container_name: mqtt-broker
    restart: unless-stopped
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - mqtt_config:/mosquitto/config
      - mqtt_data:/mosquitto/data
      - mqtt_logs:/mosquitto/log
    networks:
      - toolbox-network

networks:
  toolbox-network:
    external: true
    name: toolbox-network

volumes:
  mqtt_config:
    driver: local
  mqtt_data:
    driver: local
  mqtt_logs:
    driver: local
```

### Stack 2: uos-depthest-listener

**Stack Name:** `uos-depthest-listener`

```yaml
version: '3.8'

services:
  uos-depthest-listener:
    image: uos-depthest-listener:latest
    container_name: uos-depthest-listener
    restart: unless-stopped
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
      - MQTT_BROKER_HOST=mqtt-broker
      - MQTT_BROKER_PORT=1883
      - MQTT_BROKER_USERNAME=
      - MQTT_BROKER_PASSWORD=
    volumes:
      - uos_config:/app/config
      - uos_cache:/app/.cache
    networks:
      - toolbox-network
    depends_on:
      - mqtt-broker  # Note: This won't work across stacks

networks:
  toolbox-network:
    external: true
    name: toolbox-network

volumes:
  uos_config:
    driver: local
  uos_cache:
    driver: local
```

### Stack 3: uos-publish-json

**Stack Name:** `uos-publish-json`

```yaml
version: '3.8'

services:
  uos-publish-json:
    image: uos-publish-json:latest
    container_name: uos-publish-json
    restart: unless-stopped
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
      - MQTT_BROKER_HOST=mqtt-broker
      - MQTT_BROKER_PORT=1883
      - MQTT_BROKER_USERNAME=
      - MQTT_BROKER_PASSWORD=
    volumes:
      - publisher_config:/app/config
    networks:
      - toolbox-network

networks:
  toolbox-network:
    external: true
    name: toolbox-network

volumes:
  publisher_config:
    driver: local
```

## Environment Variables

### MQTT Broker Configuration

All services support these environment variables that override YAML configuration:

- `MQTT_BROKER_HOST` - Hostname/IP of MQTT broker (default: `mqtt-broker`)
- `MQTT_BROKER_PORT` - Port number (default: `1883`)
- `MQTT_BROKER_USERNAME` - Username for authentication (optional)
- `MQTT_BROKER_PASSWORD` - Password for authentication (optional)

### Application Configuration

- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `PYTHONUNBUFFERED` - Disable Python output buffering

## Troubleshooting

### 1. Cross-Stack Communication Issues

**Problem:** Services cannot connect to `mqtt-broker`

**Solution:**
1. Verify external network exists:
   ```bash
   docker network inspect toolbox-network
   ```

2. Check if all containers are on the same network:
   ```bash
   docker network inspect toolbox-network --format '{{range .Containers}}{{.Name}} {{end}}'
   ```

3. Test connectivity between containers:
   ```bash
   # From uos-depthest-listener container
   docker exec uos-depthest-listener ping mqtt-broker
   
   # From uos-publish-json container
   docker exec uos-publish-json ping mqtt-broker
   ```

### 2. Service Discovery Not Working

**Problem:** Container names don't resolve across stacks

**Diagnosis:**
```bash
# Check DNS resolution
docker exec uos-depthest-listener nslookup mqtt-broker
```

**Solution:**
1. Ensure all containers use the same external network
2. Verify container names are unique across stacks
3. Use IP addresses as fallback:
   ```bash
   # Get broker IP
   docker inspect mqtt-broker --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
   ```

### 3. Configuration Priority Issues

**Problem:** Environment variables not overriding YAML config

**Diagnosis:**
Check configuration summary in logs:
```bash
docker logs uos-depthest-listener | grep "environment_overrides"
```

**Solution:**
1. Verify environment variables are set in Portainer stack
2. Check variable names match exactly (case-sensitive)
3. Restart containers after configuration changes

### 4. Volume Permissions Issues

**Problem:** Containers cannot write to volumes

**Solution:**
1. Check volume ownership:
   ```bash
   docker exec mqtt-broker ls -la /mosquitto/
   ```

2. Fix permissions if needed:
   ```bash
   docker exec mqtt-broker chown -R mosquitto:mosquitto /mosquitto/
   ```

## Network Architecture Diagram

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Stack: mqtt-broker │    │Stack: uos-depthest- │    │Stack: uos-publish-  │
│                     │    │      listener       │    │      json           │
│  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │
│  │  mqtt-broker  │  │    │  │uos-depthest-  │  │    │  │uos-publish-   │  │
│  │  :1883        │  │    │  │listener       │  │    │  │json           │  │
│  │               │  │    │  │               │  │    │  │               │  │
│  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │
│                     │    │                     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       │
                           ┌─────────────────────┐
                           │  toolbox-network    │
                           │  (external)         │
                           └─────────────────────┘
```

## Best Practices

1. **Always create external network first** before deploying stacks
2. **Use consistent naming** for containers across stacks
3. **Set environment variables** for all broker connection details
4. **Monitor logs** for configuration override confirmations
5. **Test connectivity** between containers after deployment
6. **Use health checks** to ensure services are ready before dependent services start
7. **Document custom configurations** for team members