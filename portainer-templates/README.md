# Portainer Stack Templates

This directory contains Docker Compose templates for deploying the UOS Drilling Depth Estimation system as separate Portainer stacks.

## Files

- `01-mqtt-broker-stack.yml` - MQTT broker (deploy first)
- `02-uos-depthest-listener-stack.yml` - Main processing application (deploy second)
- `03-uos-publish-json-stack.yml` - Test data publisher (deploy third)
- `04-uos-publish-json-exact-stack.yml` - Exact matching test publisher (optional)

## Prerequisites

1. **Create External Network** (required before deploying any stacks):
   ```bash
   docker network create toolbox-network
   ```

2. **Build Docker Images** (if not using pre-built images):
   ```bash
   docker build -t uos-depthest-listener:latest .
   docker build -t uos-publish-json:latest .
   ```

## Deployment Order

Deploy stacks in this exact order:

1. **mqtt-broker** (first)
2. **uos-depthest-listener** (second)
3. **uos-publish-json** or **uos-publish-json-exact** (third)

## Usage in Portainer

1. Go to **Stacks** → **Add Stack**
2. Choose **Upload** and select the appropriate `.yml` file
3. Set the **Stack Name** as specified in the file comments
4. Modify environment variables as needed
5. Click **Deploy Stack**

## Configuration

### Environment Variables

All services support these environment variables:

- `MQTT_BROKER_HOST` - Hostname of MQTT broker (default: `mqtt-broker`)
- `MQTT_BROKER_PORT` - Port number (default: `1883`)
- `MQTT_BROKER_USERNAME` - Username for authentication (optional)
- `MQTT_BROKER_PASSWORD` - Password for authentication (optional)
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

### Customization

Edit the environment variables in each template file before deploying:

```yaml
environment:
  - MQTT_BROKER_HOST=your-broker-hostname
  - MQTT_BROKER_PORT=1883
  - LOG_LEVEL=DEBUG
```

## Troubleshooting

### Network Issues

If services cannot communicate:

1. Verify external network exists:
   ```bash
   docker network ls | grep toolbox-network
   ```

2. Check container connectivity:
   ```bash
   docker exec uos-depthest-listener ping mqtt-broker
   ```

### Configuration Issues

Check if environment variables are overriding YAML config:

```bash
docker logs uos-depthest-listener | grep "environment_overrides"
```

### Volume Issues

Verify volumes are created and accessible:

```bash
docker volume ls | grep mqtt
docker volume inspect mqtt_data
```

## Notes

- The `mqtt-broker` stack exposes ports 1883 and 9001 to the host
- Other stacks do not expose ports (they communicate via the internal network)
- All stacks use named volumes for persistent data
- Health checks are configured for all services
- Resource limits are set to prevent resource exhaustion