# Portainer Stack Updater

Automated update system for Portainer-managed Docker stacks in the mqtt-multistack directory.

## Overview

This script automatically detects and applies updates to Portainer stacks when:
- Docker Compose files are modified
- Source code changes require image rebuilds
- External Docker images have new versions available

## Features

- 🔄 Automatic change detection (Git-based)
- 🏗️ Local Docker image rebuilding
- 🔐 Secure Portainer API integration
- 🧪 Dry-run mode for testing
- 📝 Comprehensive logging
- 🔒 Concurrent execution prevention
- ❤️ Health checking after updates
- 🔔 Optional notifications
- 🔙 Rollback capability (planned)

## Managed Stacks

1. **mqtt-broker** - Eclipse Mosquitto MQTT broker
2. **uos-depthest-listener-cpu** - CPU-based depth estimation listener
3. **uos-publisher-json** - JSON data publisher

## Installation

1. Copy the configuration template:
   ```bash
   cp .stack-update-config.env.example .stack-update-config.env
   ```

2. Edit `.stack-update-config.env` with your Portainer credentials:
   ```bash
   nano .stack-update-config.env
   ```

3. Ensure the script is executable:
   ```bash
   chmod +x portainer-stack-updater.sh
   ```

## Configuration

### Required Settings

- `PORTAINER_URL`: Your Portainer instance URL
- `PORTAINER_USER`: Admin username
- `PORTAINER_PASS`: Admin password

### Optional Settings

- `DRY_RUN`: Test mode without making changes
- `AUTO_CONFIRM`: Skip confirmation prompts
- `ENABLE_NOTIFICATIONS`: Enable update notifications
- `LOG_FILE`: Custom log file location

## Usage

### Manual Execution

```bash
# Normal update check and apply
./portainer-stack-updater.sh

# Dry run to see what would be updated
./portainer-stack-updater.sh --dry-run

# Force update even without changes
./portainer-stack-updater.sh --force

# Auto-confirm all updates
./portainer-stack-updater.sh --auto-confirm
```

### Automated Execution (Cron)

Add to crontab for automatic updates:

```bash
# Check for updates every 30 minutes
*/30 * * * * cd /path/to/mqtt-multistack && ./portainer-stack-updater.sh --auto-confirm >> /var/log/stack-updater.log 2>&1
```

### CI/CD Integration

Call the script from your CI/CD pipeline:

```bash
# In your deployment script
ssh user@server "cd /path/to/mqtt-multistack && ./portainer-stack-updater.sh --auto-confirm"
```

## How It Works

1. **Authentication**: Connects to Portainer API and obtains JWT token
2. **Change Detection**: 
   - Checks Git status for compose file changes
   - Compares file timestamps with last update
   - Detects if source code requires image rebuilds
3. **Image Building**: Rebuilds local Docker images if needed
4. **Stack Update**: Updates stack configuration via Portainer API
5. **Health Check**: Verifies services are running after update
6. **State Tracking**: Records successful updates to prevent redundant operations

## State Management

The script maintains state in `.stack-update-state.json`:

```json
{
  "stacks": {
    "mqtt-broker": {
      "last_update": 1689123456,
      "status": "updated",
      "updated_at": "2024-07-12T10:30:00Z"
    }
  }
}
```

## Troubleshooting

### Authentication Fails
- Verify Portainer credentials in config file
- Check Portainer URL is accessible
- Ensure user has admin privileges

### Stack Not Found
- Verify stack names match exactly in Portainer
- Check EndpointId configuration

### Build Failures
- Check Docker daemon is running
- Verify build scripts exist and are executable
- Review build logs for specific errors

### Update Failures
- Check Portainer API permissions
- Verify compose file syntax
- Review Portainer logs

## Security Considerations

- Store `.stack-update-config.env` securely (it's gitignored)
- Use environment-specific credentials
- Consider using Docker secrets for production
- JWT tokens expire after 8 hours
- Lock file prevents concurrent executions

## Logs

Logs are written to `stack-updater.log` by default:

```
2024-07-12 10:30:00 [INFO] Starting Portainer Stack Updater
2024-07-12 10:30:01 [INFO] Authenticating with Portainer...
2024-07-12 10:30:02 [INFO] Checking stack: mqtt-broker
2024-07-12 10:30:03 [INFO] Git changes detected for uos-depthest-listener-cpu
2024-07-12 10:30:04 [INFO] Rebuilding Docker image: uos-depthest-listener:cpu
```

## Development

### Adding New Stacks

1. Add to the `STACKS` array in the script:
   ```bash
   STACKS["new-stack"]="new-stack/docker-compose.yml"
   ```

2. If it uses a local image, add to `LOCAL_IMAGES`:
   ```bash
   LOCAL_IMAGES["new-image:tag"]="build-new.sh"
   ```

### Custom Health Checks

Extend the `check_stack_health()` function for service-specific health checks.

### Notification Integration

Implement the `send_notification()` function for your preferred notification method.

## Known Limitations

- Rollback functionality is planned but not yet implemented
- External image update detection requires Docker Hub API integration
- Health checks are basic (container status only)

## Contributing

1. Test changes with `--dry-run` first
2. Update documentation for new features
3. Follow existing code style
4. Add error handling for new functionality

## Support

For issues or questions:
1. Check the logs first
2. Run with `--dry-run` to diagnose
3. Verify Portainer API access manually
4. Create an issue with full error details