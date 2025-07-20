# Configuration Directory

This directory provides a centralized location for project-wide configuration files and templates.

## Structure

```
config/
├── README.md                   # This file
├── templates/                  # Configuration templates
│   ├── mqtt_template.yaml      # MQTT configuration template
│   ├── docker_template.yml     # Docker compose template
│   └── deployment_template.yaml # Deployment configuration template
├── environments/               # Environment-specific configs
│   ├── development/            # Development environment
│   ├── testing/                # Testing environment
│   └── production/             # Production environment
└── schemas/                    # Configuration validation schemas
    ├── mqtt_schema.json        # MQTT config validation
    ├── docker_schema.json      # Docker config validation
    └── deployment_schema.json  # Deployment config validation
```

## Current Configuration Locations

The existing configuration files are currently located in:
- **MQTT configs**: `abyss/src/abyss/run/config/`
- **Docker configs**: Various `docker-compose*.yml` files in subdirectories
- **Deployment configs**: `mqtt-multistack/*/` directories

## Migration Plan

This centralized `config/` directory is being introduced to:

1. **Consolidate configurations** from scattered locations
2. **Provide templates** for easy setup
3. **Support environment-specific** configurations
4. **Enable validation** with JSON schemas
5. **Improve maintainability** and discoverability

## Usage

### For New Deployments
```bash
# Copy template and customize
cp config/templates/mqtt_template.yaml config/environments/development/mqtt.yaml
# Edit config/environments/development/mqtt.yaml as needed

# Validate configuration
make validate-config
```

### For Existing Configurations
Existing configurations remain functional. This directory provides:
- **Templates** for new setups
- **Validation** for existing configs  
- **Centralized reference** for all configuration options

## Validation

Use the new Makefile target to validate configurations:
```bash
make validate-config
```

This checks:
- YAML syntax in all `*.yml` and `*.yaml` files
- Docker Compose file validity
- Schema validation (when schemas are implemented)

## Best Practices

1. **Use templates** as starting points for new configurations
2. **Environment-specific** configs should inherit from templates
3. **Document changes** when modifying templates
4. **Validate before deployment** using `make validate-config`
5. **Version control** all configuration changes

## Future Enhancements

- JSON schema validation for configuration files
- Environment variable substitution in templates
- Configuration file generation scripts
- Integration with deployment automation