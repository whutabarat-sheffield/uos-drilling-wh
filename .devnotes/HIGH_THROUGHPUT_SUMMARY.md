# High-Throughput Architecture - Quick Reference

## Current State
- **Performance**: 80-100 messages/second
- **Architecture**: Single-threaded, in-memory buffers
- **Issues**: Message drops, no recovery, can't scale

## Target State
- **Performance**: 1,000-100,000+ messages/second
- **Architecture**: Multi-tier, distributed, auto-scaling
- **Benefits**: Zero message loss, automatic recovery, horizontal scaling

## Implementation Phases

### Phase 1: Async Processing (Weeks 1-3)
- Convert to asyncio architecture
- Multiple MQTT clients (10x parallel)
- Target: 1,000 msg/sec
- Files: `drilling_analyser.py`, `message_buffer.py`

### Phase 2: RabbitMQ (Weeks 4-6)
- Add persistent queuing
- Decouple ingestion from processing
- Target: 5,000 msg/sec
- New: Dead letter queues, worker pools

### Phase 3: Auto-scaling (Weeks 7-9)
- Dynamic worker scaling
- Batch processing for bursts
- Target: 10,000+ msg/sec
- New: Monitoring, circuit breakers

## Key Architecture Changes

```
Current:
MQTT → Buffer → Single Thread → Publisher

Future:
MQTT → Async Ingestion → RabbitMQ/Kafka → Worker Pool → Publisher
         (10 clients)      (persistent)     (auto-scale)
```

## Quick Start Commands

```bash
# Check current performance
python stress_test.py --rate 100 --duration 60

# Enable high-throughput mode
# In mqtt_conf.yaml:
high_throughput:
  enabled: true
  architecture: "async"  # or "rabbitmq", "hybrid"

# Run with new architecture
python uos_depth_est_mqtt.py --config mqtt_conf_ht.yaml

# Monitor performance
curl http://localhost:9090/metrics | grep drilling_
```

## Performance Comparison

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Throughput | 100/s | 1,000/s | 5,000/s | 10,000+/s |
| Message Loss | >5% | <1% | 0% | 0% |
| Recovery | None | Limited | Full | Full |
| Scaling | None | Vertical | Manual | Automatic |

## When to Use Each Architecture

- **Async (Phase 1)**: For immediate 10x improvement
- **RabbitMQ (Phase 2)**: When you need persistence
- **Hybrid (Phase 3)**: For production workloads
- **Kafka**: For 100,000+ msg/sec

## Migration Checklist

- [ ] Benchmark current system
- [ ] Deploy monitoring
- [ ] Test async architecture
- [ ] Deploy RabbitMQ
- [ ] Implement gradual rollout
- [ ] Monitor and optimize

## Key Files for Implementation

1. `drilling_analyser.py` - Main orchestrator
2. `message_buffer.py` - Buffer management
3. `client_manager.py` - MQTT connections
4. `async_processor.py` - New async engine
5. `rabbitmq_bridge.py` - Queue integration

## Success Metrics

- Messages/second > 1000
- Drop rate < 0.1%
- P99 latency < 200ms
- Zero message loss
- Auto-recovery from failures

## Contact & Resources

- Full documentation: `.devnotes/HIGH_THROUGHPUT_ARCHITECTURE.md`
- Performance tests: `tests/performance/`
- Monitoring dashboard: `http://localhost:9090`
- RabbitMQ admin: `http://localhost:15672`