# High-Throughput Message Processing Architecture for UOS Drilling

## Executive Summary

### Problem Statement
The current UOS Drilling system processes MQTT messages at approximately 80-100 messages per second, with a target of 1000+ messages per second. Users have reported:
- Messages being dropped during high-volume scenarios
- Lack of visibility into buffer capacity and system stress
- No recovery mechanism for dropped messages
- Single-threaded processing creating bottlenecks

### Proposed Solution
Transform the architecture from a single-threaded, in-memory buffer system to a multi-tier, scalable architecture capable of handling 1000-100,000+ messages per second with automatic scaling, persistence, and fault tolerance.

### Expected Benefits
- **10-1000x throughput improvement** (from 100 to 100,000+ msg/sec)
- **Zero message loss** with persistent queuing
- **Automatic scaling** based on load
- **Fault tolerance** with message replay capabilities
- **Real-time monitoring** of system health
- **Gradual migration path** with backward compatibility

## Current Architecture Analysis

### System Overview
```
MQTT Broker → MessageBuffer (in-memory) → Single Processing Thread → Result Publisher
```

### Bottlenecks and Limitations

1. **Single Processing Thread**
   - Location: `DrillingDataAnalyser.continuous_processing()` 
   - Issue: Sequential processing limits throughput
   - Impact: ~100 msg/sec maximum

2. **In-Memory Buffers**
   - Location: `MessageBuffer` class with `Dict[str, List[TimestampedData]]`
   - Issue: Volatile storage, fixed size limits
   - Impact: Message loss on overflow or crash

3. **Tight Coupling**
   - Issue: Message reception directly tied to processing
   - Impact: Can't scale components independently

4. **No Horizontal Scaling**
   - Issue: Can't add more workers
   - Impact: Limited by single machine capacity

### Current Performance Metrics
- **Throughput**: 80-100 messages/second
- **Latency**: <100ms average
- **Buffer Capacity**: 10,000 messages default
- **Drop Rate**: >5% under heavy load
- **Recovery**: None - dropped messages are lost

### Why Change is Needed
1. **Business Requirements**: Target of 1000+ signals/second for stress testing
2. **Reliability**: Current system drops messages silently after warnings
3. **Scalability**: No path to handle future growth
4. **Observability**: Limited visibility into system performance
5. **Recovery**: No mechanism to replay failed messages

## Proposed Architectures

### 1. RabbitMQ-Based Decoupled Architecture

**Design:**
```
MQTT → Ingestion Service → RabbitMQ → Worker Pool → Result Publisher
                               ↓
                          Dead Letter Queue
```

**Benefits:**
- Message persistence across restarts
- Built-in flow control and backpressure
- Dynamic worker scaling
- Dead letter queues for failed messages
- Management UI for monitoring

**Implementation Example:**
```python
# Ingestion Service
class RabbitMQIngestion:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('rabbitmq-server')
        )
        self.channel = self.connection.channel()
        
        # Declare exchanges and queues
        self.channel.exchange_declare(
            exchange='drilling_data',
            exchange_type='topic',
            durable=True
        )
        
        # Queues for different message types
        for msg_type in ['result', 'trace', 'heads']:
            self.channel.queue_declare(
                queue=f'drilling_{msg_type}',
                durable=True,
                arguments={
                    'x-max-length': 100000,
                    'x-overflow': 'reject-publish-dlx',
                    'x-dead-letter-exchange': 'drilling_dlx'
                }
            )
    
    async def mqtt_to_rabbitmq(self):
        """Forward MQTT messages to RabbitMQ with zero processing"""
        async with aiomqtt.Client("mqtt-broker") as client:
            await client.subscribe("OPCPUBSUB/+/+/#")
            
            async for message in client.messages:
                routing_key = self.extract_routing_key(message.topic)
                self.channel.basic_publish(
                    exchange='drilling_data',
                    routing_key=routing_key,
                    body=message.payload,
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Persistent
                        timestamp=int(time.time()),
                        headers={'mqtt_topic': message.topic}
                    )
                )

# Worker Pool
class DepthEstimationWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.correlator = MessageCorrelator()
        self.depth_inference = DepthInference()
        
    def start(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('rabbitmq-server')
        )
        channel = connection.channel()
        
        # Consume from multiple queues
        channel.basic_qos(prefetch_count=10)
        channel.basic_consume(
            queue='drilling_result',
            on_message_callback=self.process_message
        )
        channel.basic_consume(
            queue='drilling_trace',
            on_message_callback=self.process_message
        )
        
        channel.start_consuming()
    
    def process_message(self, ch, method, properties, body):
        try:
            # Correlate and process
            if self.correlator.can_correlate(properties.headers):
                correlated_data = self.correlator.get_correlated_messages()
                result = self.depth_inference.process(correlated_data)
                self.publish_result(result)
            
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            # Send to dead letter queue
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
```

**Performance**: 5,000+ msg/sec with proper tuning

### 2. Async Stream Processing Architecture

**Design:**
```python
class AsyncStreamProcessor:
    def __init__(self):
        # High-performance async queues
        self.message_streams = {
            'result': asyncio.Queue(maxsize=10000),
            'trace': asyncio.Queue(maxsize=10000),
            'heads': asyncio.Queue(maxsize=10000)
        }
        self.correlation_cache = TTLCache(maxsize=10000, ttl=30)
        
    async def run(self):
        """Main entry point - runs all components concurrently"""
        await asyncio.gather(
            # 10 parallel MQTT ingestion workers
            *[self.mqtt_ingestion_worker(i) for i in range(10)],
            # 20 parallel correlation workers
            *[self.correlation_worker(i) for i in range(20)],
            # 5 result publishers
            *[self.result_publisher(i) for i in range(5)],
            # Monitoring task
            self.monitor_performance()
        )
    
    async def mqtt_ingestion_worker(self, worker_id):
        """High-speed MQTT message ingestion"""
        async with aiomqtt.Client(
            hostname="mqtt-broker",
            client_id=f"ingestion-{worker_id}",
            clean_session=False  # Persist subscription
        ) as client:
            await client.subscribe("OPCPUBSUB/+/+/#", qos=1)
            
            async for message in client.messages:
                # Parse topic to determine stream
                stream_type = self.get_stream_type(message.topic)
                
                # Non-blocking put with overflow handling
                try:
                    self.message_streams[stream_type].put_nowait({
                        'topic': message.topic,
                        'payload': message.payload,
                        'timestamp': time.time()
                    })
                except asyncio.QueueFull:
                    logging.warning(f"Queue full for {stream_type}")
                    # Could implement overflow to disk here
    
    async def correlation_worker(self, worker_id):
        """Correlate messages across streams"""
        while True:
            try:
                # Try to get messages from each stream
                result_msg = await asyncio.wait_for(
                    self.message_streams['result'].get(), 
                    timeout=0.1
                )
                
                # Extract correlation key
                tool_key = self.extract_tool_key(result_msg['topic'])
                
                # Check cache for correlated messages
                if tool_key in self.correlation_cache:
                    trace_msg = self.correlation_cache[tool_key]['trace']
                    heads_msg = self.correlation_cache[tool_key].get('heads')
                    
                    # Process correlated messages
                    await self.process_correlated_messages(
                        result_msg, trace_msg, heads_msg
                    )
                else:
                    # Store for future correlation
                    self.correlation_cache[tool_key] = {'result': result_msg}
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Correlation error: {e}")
    
    async def process_correlated_messages(self, result, trace, heads):
        """Process messages with async depth estimation"""
        # CPU-bound work in thread pool
        loop = asyncio.get_event_loop()
        depth_result = await loop.run_in_executor(
            None,  # Default thread pool
            self.depth_inference.estimate,
            result, trace, heads
        )
        
        # Queue for publishing
        await self.publish_queue.put(depth_result)
```

**Performance**: 1,000-5,000 msg/sec with low latency

### 3. Kubernetes Job-Based Architecture

**Design:**
```yaml
# Message Accumulator Service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: message-accumulator
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: accumulator
        image: drilling/accumulator:latest
        env:
        - name: BATCH_SIZE
          value: "1000"
        - name: BATCH_TIMEOUT
          value: "5"
        - name: S3_BUCKET
          value: "drilling-batches"

# Job Spawner Service
apiVersion: v1
kind: ConfigMap
metadata:
  name: job-template
data:
  job.yaml: |
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: depth-estimation-{{.BatchID}}
    spec:
      parallelism: 10
      completions: 10
      backoffLimit: 3
      template:
        spec:
          containers:
          - name: processor
            image: drilling/depth-processor:latest
            resources:
              requests:
                memory: "2Gi"
                cpu: "1"
              limits:
                memory: "4Gi"
                cpu: "2"
            env:
            - name: BATCH_ID
              value: "{{.BatchID}}"
            - name: S3_BUCKET
              value: "drilling-batches"
          restartPolicy: OnFailure
```

**Implementation:**
```python
class BatchAccumulator:
    def __init__(self):
        self.batch_size = 1000
        self.batch_timeout = 5.0
        self.current_batch = []
        self.s3_client = boto3.client('s3')
        self.k8s_client = kubernetes.client.BatchV1Api()
        
    async def accumulate_messages(self):
        """Accumulate messages into batches"""
        async with aiomqtt.Client("mqtt-broker") as client:
            await client.subscribe("OPCPUBSUB/+/+/#")
            
            batch_start = time.time()
            
            async for message in client.messages:
                self.current_batch.append({
                    'topic': message.topic,
                    'payload': message.payload.decode(),
                    'timestamp': time.time()
                })
                
                # Check batch triggers
                if (len(self.current_batch) >= self.batch_size or 
                    time.time() - batch_start > self.batch_timeout):
                    await self.flush_batch()
                    batch_start = time.time()
    
    async def flush_batch(self):
        """Save batch to S3 and spawn processing job"""
        if not self.current_batch:
            return
            
        batch_id = str(uuid.uuid4())
        
        # Save to S3
        self.s3_client.put_object(
            Bucket='drilling-batches',
            Key=f'batch-{batch_id}.json',
            Body=json.dumps(self.current_batch)
        )
        
        # Spawn Kubernetes job
        job_manifest = self.render_job_template(batch_id)
        self.k8s_client.create_namespaced_job(
            namespace='default',
            body=job_manifest
        )
        
        logging.info(f"Spawned job for batch {batch_id} with {len(self.current_batch)} messages")
        self.current_batch = []
```

**Performance**: 10,000+ msg/sec with batch processing

### 4. Apache Kafka + Kafka Streams

**Design:**
```java
// Kafka Streams Topology
public class DepthEstimationTopology {
    public static Topology build() {
        StreamsBuilder builder = new StreamsBuilder();
        
        // Input streams with deserialization
        KStream<String, DrillMessage> results = builder.stream(
            "drilling-results",
            Consumed.with(Serdes.String(), drillMessageSerde())
        );
        
        KStream<String, DrillMessage> traces = builder.stream(
            "drilling-traces", 
            Consumed.with(Serdes.String(), drillMessageSerde())
        );
        
        // Window-based correlation join
        KStream<String, CorrelatedData> correlated = results
            .join(
                traces,
                (result, trace) -> new CorrelatedData(result, trace),
                JoinWindows.of(Duration.ofSeconds(30))
                    .grace(Duration.ofSeconds(10)),
                StreamJoined.with(
                    Serdes.String(),
                    drillMessageSerde(),
                    drillMessageSerde()
                )
            );
        
        // Stateful depth estimation
        KStream<String, DepthEstimation> estimations = correlated
            .transformValues(
                () -> new DepthEstimationTransformer(),
                "depth-estimation-store"
            );
        
        // Output to results topic
        estimations.to(
            "depth-estimations",
            Produced.with(Serdes.String(), depthEstimationSerde())
        );
        
        return builder.build();
    }
}
```

**Python Integration:**
```python
from confluent_kafka import Producer, Consumer
import json

class KafkaIntegration:
    def __init__(self):
        self.producer = Producer({
            'bootstrap.servers': 'kafka-cluster:9092',
            'compression.type': 'snappy',
            'linger.ms': 10,
            'batch.size': 65536
        })
        
    async def mqtt_to_kafka_bridge(self):
        """High-performance MQTT to Kafka bridge"""
        async with aiomqtt.Client("mqtt-broker") as client:
            await client.subscribe("OPCPUBSUB/+/+/#")
            
            async for message in client.messages:
                # Determine Kafka topic and key
                kafka_topic = self.get_kafka_topic(message.topic)
                kafka_key = self.extract_tool_key(message.topic)
                
                # Async produce to Kafka
                self.producer.produce(
                    topic=kafka_topic,
                    key=kafka_key,
                    value=message.payload,
                    callback=self.delivery_report
                )
                
                # Periodic flush for batching
                if random.random() < 0.01:  # 1% chance
                    self.producer.flush(timeout=0.1)
```

**Performance**: 100,000+ msg/sec with proper cluster sizing

### 5. Hybrid Multi-Tier Architecture (Recommended)

**Design:**
```
Tier 1: Fast Ingestion
├── 10x Async MQTT Clients (aiomqtt)
├── Zero processing overhead
└── Publishes to Tier 2

Tier 2: Durable Queue  
├── RabbitMQ for <10k msg/sec
├── Kafka for >10k msg/sec
├── Auto-switching based on load
└── Persistence and replay

Tier 3: Adaptive Processing
├── Async workers for real-time
├── Batch jobs for bulk processing  
├── GPU workers for complex inference
└── Auto-scaling based on queue depth
```

**Implementation:**
```python
class HybridArchitecture:
    def __init__(self):
        self.ingestion_tier = IngestionTier()
        self.queue_tier = QueueTier()
        self.processing_tier = ProcessingTier()
        self.monitoring = MonitoringSystem()
        
    async def start(self):
        await asyncio.gather(
            self.ingestion_tier.start(),
            self.queue_tier.start(),
            self.processing_tier.start(),
            self.monitoring.start()
        )

class IngestionTier:
    def __init__(self):
        self.mqtt_clients = []
        self.message_rate = 0
        self.queue_backend = 'rabbitmq'  # or 'kafka'
        
    async def start(self):
        # Create 10 parallel MQTT clients
        for i in range(10):
            client = MQTTIngestionClient(
                client_id=f"ingestion-{i}",
                queue_backend=self.queue_backend
            )
            self.mqtt_clients.append(client)
            
        await asyncio.gather(*[c.run() for c in self.mqtt_clients])
    
    async def adaptive_backend_selection(self):
        """Switch between RabbitMQ and Kafka based on load"""
        while True:
            await asyncio.sleep(10)
            
            if self.message_rate > 10000 and self.queue_backend == 'rabbitmq':
                logging.info("Switching to Kafka due to high load")
                await self.switch_to_kafka()
            elif self.message_rate < 5000 and self.queue_backend == 'kafka':
                logging.info("Switching to RabbitMQ for lower load")
                await self.switch_to_rabbitmq()

class ProcessingTier:
    def __init__(self):
        self.async_workers = []
        self.batch_processor = None
        self.gpu_workers = []
        
    async def start(self):
        # Start with minimal workers
        await self.scale_async_workers(5)
        
        # Monitor and auto-scale
        asyncio.create_task(self.auto_scale())
    
    async def auto_scale(self):
        """Adaptive scaling based on queue depth"""
        while True:
            await asyncio.sleep(5)
            
            metrics = await self.get_queue_metrics()
            queue_depth = metrics['queue_depth']
            processing_rate = metrics['processing_rate']
            
            if queue_depth > 10000:
                # Spawn batch processing job
                await self.spawn_batch_job(size=5000)
            elif queue_depth > 1000:
                # Scale up async workers
                await self.scale_async_workers(20)
            elif queue_depth < 100:
                # Scale down to save resources
                await self.scale_async_workers(5)
            
            # GPU workers for complex scenarios
            if metrics['complexity_score'] > 0.8:
                await self.scale_gpu_workers(2)
```

## Recommended Implementation Approach

### Phase 1: Async Stream Processing (Weeks 1-3)

**Goal**: Achieve 1000 msg/sec with minimal changes

**Tasks**:
1. Refactor `DrillingDataAnalyser` to use asyncio
2. Replace single processing thread with worker pool
3. Implement async MQTT clients with aiomqtt
4. Add performance monitoring

**Key Files to Modify**:
- `/abyss/src/abyss/mqtt/components/drilling_analyser.py`
- `/abyss/src/abyss/mqtt/components/message_buffer.py`
- `/abyss/src/abyss/mqtt/components/client_manager.py`

### Phase 2: RabbitMQ Integration (Weeks 4-6)

**Goal**: Add persistence and achieve 5000 msg/sec

**Tasks**:
1. Deploy RabbitMQ with clustering
2. Create ingestion service
3. Implement worker pool
4. Add dead letter queue handling
5. Parallel execution with existing system

**New Components**:
- `rabbitmq_ingestion.py` - MQTT to RabbitMQ bridge
- `rabbitmq_worker.py` - Processing workers
- `rabbitmq_monitor.py` - Queue monitoring

### Phase 3: Auto-scaling and Optimization (Weeks 7-9)

**Goal**: Dynamic scaling and 10,000+ msg/sec capability

**Tasks**:
1. Implement queue depth monitoring
2. Add worker auto-scaling logic
3. Create batch processing pipeline
4. Add circuit breakers and backpressure
5. Implement monitoring dashboard

## Technical Implementation Details

### Dependencies to Add

```toml
# In pyproject.toml or requirements.txt
aiomqtt = "^2.0.0"          # Async MQTT client
pika = "^1.3.0"             # RabbitMQ client
aiokafka = "^0.8.0"         # Kafka client (future)
prometheus-client = "^0.15.0" # Metrics
aioredis = "^2.0.0"         # Caching
asyncio-throttle = "^1.0.0"  # Rate limiting
```

### Configuration Schema

```yaml
# Enhanced mqtt_conf.yaml
mqtt:
  broker:
    host: "localhost"
    port: 1883
  
  # New high-throughput settings
  high_throughput:
    enabled: true
    architecture: "hybrid"  # async, rabbitmq, kafka, hybrid
    
    ingestion:
      workers: 10
      max_connections: 50
      
    queue:
      backend: "rabbitmq"  # or kafka
      rabbitmq:
        host: "localhost"
        port: 5672
        exchange: "drilling_data"
        queue_size: 100000
      kafka:
        brokers: ["localhost:9092"]
        topics:
          results: "drilling-results"
          traces: "drilling-traces"
    
    processing:
      async_workers: 20
      batch_size: 1000
      batch_timeout: 5
      max_workers: 100
      scale_up_threshold: 1000  # queue depth
      scale_down_threshold: 100
    
    monitoring:
      enabled: true
      port: 9090
      metrics_interval: 10
```

### Backward Compatibility Strategy

```python
class BackwardCompatibleAnalyser:
    """Wrapper to maintain API compatibility"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        if self.config.get('high_throughput', {}).get('enabled', False):
            # Use new architecture
            self.implementation = HybridArchitecture(config_path)
        else:
            # Use legacy architecture
            self.implementation = DrillingDataAnalyser(config_path)
    
    def run(self):
        """Same interface, different implementation"""
        if asyncio.iscoroutinefunction(self.implementation.run):
            asyncio.run(self.implementation.run())
        else:
            self.implementation.run()
```

### Testing Strategy

```python
# Performance test harness
class PerformanceTestHarness:
    async def test_throughput(self, architecture, target_rate):
        """Test if architecture meets throughput targets"""
        publisher = StressTestPublisher(rate=target_rate)
        consumer = TestConsumer(architecture)
        
        # Run for 60 seconds
        await asyncio.gather(
            publisher.run(duration=60),
            consumer.run(duration=60)
        )
        
        # Verify results
        assert consumer.messages_received >= target_rate * 60 * 0.95
        assert consumer.drop_rate < 0.01
        assert consumer.avg_latency < 200  # ms

# Integration tests
async def test_rabbitmq_failover():
    """Test RabbitMQ connection failure handling"""
    architecture = HybridArchitecture(test_config)
    await architecture.start()
    
    # Simulate RabbitMQ failure
    await stop_rabbitmq()
    await asyncio.sleep(5)
    
    # Should switch to fallback
    assert architecture.queue_tier.is_using_fallback()
    
    # Restore RabbitMQ
    await start_rabbitmq()
    await asyncio.sleep(10)
    
    # Should recover
    assert not architecture.queue_tier.is_using_fallback()
```

## Migration Plan

### Pre-migration Checklist
- [ ] Load test current system to establish baseline
- [ ] Set up monitoring infrastructure
- [ ] Deploy RabbitMQ/Kafka in test environment
- [ ] Create rollback procedures
- [ ] Document current message formats

### Migration Steps

1. **Parallel Deployment** (Week 1)
   - Deploy new architecture alongside existing
   - Route 1% of traffic to new system
   - Monitor for issues

2. **Gradual Rollout** (Weeks 2-3)
   - Increase traffic: 1% → 10% → 50% → 100%
   - Monitor performance at each stage
   - Fix issues as they arise

3. **Optimization** (Week 4)
   - Tune based on production metrics
   - Optimize resource usage
   - Document lessons learned

### Rollback Plan
```bash
#!/bin/bash
# Quick rollback script
kubectl set image deployment/drilling-analyser \
  drilling-analyser=drilling-analyser:legacy

# Or via feature flag
curl -X POST http://control-plane/api/features \
  -d '{"high_throughput_enabled": false}'
```

## Monitoring and Success Metrics

### Key Performance Indicators (KPIs)

1. **Throughput Metrics**
   - Messages processed per second
   - Queue ingestion rate
   - Processing completion rate

2. **Latency Metrics**
   - End-to-end message latency (p50, p95, p99)
   - Queue time
   - Processing time

3. **Reliability Metrics**
   - Message drop rate
   - Dead letter queue size
   - Error rate by type

4. **Resource Metrics**
   - CPU usage by component
   - Memory usage
   - Network bandwidth
   - Queue depths

### Monitoring Implementation

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics collectors
messages_received = Counter('drilling_messages_received_total', 
                          'Total messages received', 
                          ['message_type'])
                          
messages_processed = Counter('drilling_messages_processed_total',
                           'Total messages processed',
                           ['status'])
                           
processing_duration = Histogram('drilling_processing_duration_seconds',
                               'Message processing duration',
                               buckets=[.001, .01, .1, .5, 1.0, 5.0])
                               
queue_depth = Gauge('drilling_queue_depth',
                   'Current queue depth',
                   ['queue_name'])

# Dashboard queries
DASHBOARD_QUERIES = {
    'throughput': 'rate(drilling_messages_processed_total[1m])',
    'latency_p99': 'histogram_quantile(0.99, drilling_processing_duration_seconds)',
    'drop_rate': 'rate(drilling_messages_dropped_total[5m]) / rate(drilling_messages_received_total[5m])',
    'queue_saturation': 'drilling_queue_depth / drilling_queue_capacity'
}
```

## Future Considerations

### Scaling Beyond 100k msg/sec
1. **Sharding Strategy**: Partition by tool_id
2. **Geographic Distribution**: Multi-region deployment
3. **GPU Acceleration**: For complex depth inference
4. **Edge Processing**: Process at data source

### Advanced Features
1. **Machine Learning Pipeline**: Real-time model updates
2. **Anomaly Detection**: Identify unusual patterns
3. **Predictive Scaling**: Scale before load arrives
4. **Data Lake Integration**: Long-term storage and analysis

### Technology Evolution
1. **MQTT 5.0**: Shared subscriptions for load balancing
2. **HTTP/3**: For result publishing
3. **WebAssembly**: For edge processing
4. **Kubernetes Operators**: Automated management

## Conclusion

This architecture transformation provides a clear path from the current 100 msg/sec system to a 100,000+ msg/sec platform. The phased approach allows gradual adoption with minimal risk, while the monitoring infrastructure ensures visibility throughout the migration.

Key success factors:
- Start simple with async processing
- Add complexity only when needed
- Monitor everything from day one
- Plan for 10x growth at each phase
- Maintain backward compatibility

The investment in this architecture will position the UOS Drilling platform for years of growth while maintaining reliability and performance.