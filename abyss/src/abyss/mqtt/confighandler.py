"""MQTT Configuration Handler for the Drilling Data Analysis System."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


@dataclass
class BrokerConfig:
    """MQTT broker connection settings."""
    
    host: str
    port: int
    username: str = ""
    password: str = ""


@dataclass
class ListenerConfig:
    """MQTT topic structure configuration for listeners."""
    
    root: str
    toolboxid: str
    toolid: str
    result: str
    trace: str
    
    def get_result_topic(self) -> str:
        """Get the complete topic string for result messages."""
        return f"{self.root}/{self.toolboxid}/{self.toolid}/{self.result}"
    
    def get_trace_topic(self) -> str:
        """Get the complete topic string for trace messages."""
        return f"{self.root}/{self.toolboxid}/{self.toolid}/{self.trace}"


@dataclass
class DataIdsConfig:
    """Configuration for data field mappings in MQTT messages."""
    
    machine_id: str
    result_id: str
    trace_result_id: str
    position: str
    thrust: str
    torque: str
    step: str
    torque_empty_vals: str
    thrust_empty_vals: str
    step_vals: str


@dataclass
class EstimationConfig:
    """Configuration for estimation data paths."""
    
    keypoints: str
    depth_estimation: str


@dataclass
class PublisherConfig:
    """Configuration for test publisher."""
    
    topics: Dict[str, str]
    client_id: str


@dataclass
class MQTTConfig:
    """MQTT Configuration for the Drilling Data Analysis System."""
    
    broker: BrokerConfig
    listener: ListenerConfig
    data_ids: DataIdsConfig
    estimation: EstimationConfig
    publisher: PublisherConfig


@dataclass
class ConfigHandler:
    """Configuration handler for the MQTT-based Drilling Data Analysis System."""
    
    mqtt: MQTTConfig
    _config_path: Optional[Path] = field(default=None, repr=False)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "ConfigHandler":
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
            
        Returns:
            A ConfigHandler instance with the loaded configuration
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        broker_data = config_data["mqtt"]["broker"]
        broker_config = BrokerConfig(
            host=broker_data["host"],
            port=broker_data["port"],
            username=broker_data.get("username", ""),
            password=broker_data.get("password", "")
        )
        
        listener_data = config_data["mqtt"]["listener"]
        listener_config = ListenerConfig(
            root=listener_data["root"],
            toolboxid=listener_data["toolboxid"],
            toolid=listener_data["toolid"],
            result=listener_data["result"],
            trace=listener_data["trace"]
        )
        
        data_ids_data = config_data["mqtt"]["data_ids"]
        data_ids_config = DataIdsConfig(
            machine_id=data_ids_data["machine_id"],
            result_id=data_ids_data["result_id"],
            trace_result_id=data_ids_data["trace_result_id"],
            position=data_ids_data["position"],
            thrust=data_ids_data["thrust"],
            torque=data_ids_data["torque"],
            step=data_ids_data["step"],
            torque_empty_vals=data_ids_data["torque_empty_vals"],
            thrust_empty_vals=data_ids_data["thrust_empty_vals"],
            step_vals=data_ids_data["step_vals"]
        )
        
        estimation_data = config_data["mqtt"]["estimation"]
        estimation_config = EstimationConfig(
            keypoints=estimation_data["keypoints"],
            depth_estimation=estimation_data["depth_estimation"]
        )
        
        publisher_data = config_data["mqtt"]["publisher"]
        publisher_config = PublisherConfig(
            topics=publisher_data["topics"],
            client_id=publisher_data["client_id"]
        )
        
        mqtt_config = MQTTConfig(
            broker=broker_config,
            listener=listener_config,
            data_ids=data_ids_config,
            estimation=estimation_config,
            publisher=publisher_config
        )
        
        handler = cls(mqtt=mqtt_config)
        handler._config_path = config_path
        return handler
    
    def get_broker_url(self) -> str:
        """Get the broker URL in format 'host:port'."""
        return f"{self.mqtt.broker.host}:{self.mqtt.broker.port}"
    
    def get_listen_topics(self) -> List[str]:
        """Get all topics that should be subscribed to."""
        return [
            self.mqtt.listener.get_result_topic(),
            self.mqtt.listener.get_trace_topic()
        ]
    
    def get_data_path(self, data_type: str) -> str:
        """Get the JSON path for a specific data type.
        
        Args:
            data_type: The type of data to retrieve (e.g., 'position', 'thrust')
            
        Returns:
            The JSON path string for the specified data type
        
        Raises:
            KeyError: If the data type is not configured
        """
        if hasattr(self.mqtt.data_ids, data_type):
            return getattr(self.mqtt.data_ids, data_type)
        raise KeyError(f"Unknown data type: {data_type}")
    
    def is_trace_topic(self, topic: str) -> bool:
        """Check if a topic is a trace topic.
        
        Args:
            topic: The full topic string to check
            
        Returns:
            True if the topic matches the trace topic pattern
        """
        parts = topic.split('/')
        if len(parts) < 4:
            return False
        
        # Check if the last part or last parts combined match the trace pattern
        trace_parts = self.mqtt.listener.trace.split('/')
        if len(trace_parts) == 1:
            return parts[-1] == trace_parts[0]
        return '/'.join(parts[-len(trace_parts):]) == self.mqtt.listener.trace
    
    def is_result_topic(self, topic: str) -> bool:
        """Check if a topic is a result topic.
        
        Args:
            topic: The full topic string to check
            
        Returns:
            True if the topic matches the result topic pattern
        """
        parts = topic.split('/')
        if len(parts) < 4:
            return False
        
        # Check if the last part matches the result pattern
        return parts[-1] == self.mqtt.listener.result
