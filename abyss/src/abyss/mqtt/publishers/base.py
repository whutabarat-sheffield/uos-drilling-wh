"""
Base MQTT Publisher functionality shared across all publisher modes.
"""

import csv
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import paho.mqtt.client as mqtt
import yaml

from .config_loader import PublisherConfigLoader, load_publisher_config


@dataclass
class PublisherConfig:
    """Configuration for MQTT publishers."""
    broker_host: str = "localhost"
    broker_port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Topic configuration
    root_topic: str = "OPCPUBSUB"
    result_suffix: str = "ResultManagement"
    trace_suffix: str = "ResultManagement/Trace"
    heads_suffix: str = "AssetManagement/Heads"
    
    # Publishing parameters
    sleep_min: float = 0.1
    sleep_max: float = 0.3
    repetitions: int = 10
    
    # Signal tracking
    track_signals: bool = False
    signal_log: str = "sent_signals.csv"
    
    # Test data
    test_data_path: Optional[Path] = None
    
    # Tool identifiers
    toolbox_ids: List[str] = field(default_factory=lambda: [
        'ILLL502033771', 'ILLL502033772', 'ILLL502033773',
        'ILLL502033774', 'ILLL502033775'
    ])
    tool_ids: List[str] = field(default_factory=lambda: [
        'setitec001', 'setitec002', 'setitec003',
        'setitec004', 'setitec005'
    ])
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PublisherConfig':
        """Load configuration from YAML file."""
        config = load_publisher_config(config_path)
        
        return cls(
            broker_host=config['broker_host'],
            broker_port=config['broker_port'],
            username=config['username'],
            password=config['password'],
            root_topic=config['root_topic'],
            result_suffix=config['result_suffix'],
            trace_suffix=config['trace_suffix'],
            heads_suffix=config['heads_suffix'],
        )


class BasePublisher(ABC):
    """Base class for all MQTT publishers."""
    
    def __init__(self, config: PublisherConfig):
        self.config = config
        self.client: Optional[mqtt.Client] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Signal tracking
        self._signal_counter = 0
        self._start_time = None
        
        # Test data cache
        self._test_data_cache: Dict[str, Tuple[str, str, str, str]] = {}
    
    def setup_client(self) -> mqtt.Client:
        """Set up and connect MQTT client."""
        self.client = mqtt.Client()
        
        if self.config.username and self.config.password:
            self.client.username_pw_set(self.config.username, self.config.password)
        
        try:
            self.client.connect(self.config.broker_host, self.config.broker_port)
            self.logger.info(f"Connected to MQTT broker at {self.config.broker_host}:{self.config.broker_port}")
            return self.client
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.disconnect()
            self.logger.info("Disconnected from MQTT broker")
    
    def find_test_data_folders(self, validate: bool = False) -> List[Path]:
        """Find all folders containing test data.
        
        Args:
            validate: If True, validate JSON files during discovery
            
        Returns:
            List of valid test data folders
        """
        if not self.config.test_data_path:
            raise ValueError("No test data path configured")
        
        path = self.config.test_data_path
        
        if not path.exists():
            raise ValueError(f"Test data path does not exist: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Test data path is not a directory: {path}")
        
        # Look for subdirectories with JSON files
        data_folders = []
        invalid_folders = []
        
        for subdir in path.iterdir():
            if subdir.is_dir():
                json_files = list(subdir.glob("*.json"))
                if json_files:
                    # Check for required files
                    required_files = ["ResultManagement.json", "Trace.json", "Heads.json"]
                    found_files = {f.name for f in json_files}
                    
                    if all(req in found_files for req in required_files):
                        # Optionally validate JSON content
                        if validate:
                            try:
                                for req_file in required_files:
                                    with open(subdir / req_file) as f:
                                        json.load(f)  # Just validate, don't keep in memory
                                data_folders.append(subdir)
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Invalid JSON in {subdir}: {e}")
                                invalid_folders.append(subdir)
                            except Exception as e:
                                self.logger.warning(f"Error validating {subdir}: {e}")
                                invalid_folders.append(subdir)
                        else:
                            data_folders.append(subdir)
                    else:
                        self.logger.warning(f"Folder {subdir} missing required files: {set(required_files) - found_files}")
        
        if not data_folders:
            # Check if the directory itself contains the files
            json_files = list(path.glob("*.json"))
            if json_files:
                required_files = ["ResultManagement.json", "Trace.json", "Heads.json"]
                found_files = {f.name for f in json_files}
                
                if all(req in found_files for req in required_files):
                    if validate:
                        try:
                            for req_file in required_files:
                                with open(path / req_file) as f:
                                    json.load(f)
                            data_folders.append(path)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON in {path}: {e}")
                            invalid_folders.append(path)
                    else:
                        data_folders.append(path)
        
        if not data_folders:
            raise ValueError(f"No valid test data folders found in {path}")
        
        if invalid_folders:
            self.logger.warning(f"Skipped {len(invalid_folders)} folders with invalid JSON: {invalid_folders}")
        
        self.logger.info(f"Found {len(data_folders)} valid test data folders")
        return sorted(data_folders)
    
    def load_test_data(self, folder: Path) -> Tuple[str, str, str, str]:
        """Load test data from a folder, using cache if available."""
        folder_key = str(folder)
        
        if folder_key in self._test_data_cache:
            return self._test_data_cache[folder_key]
        
        # Load ResultManagement.json
        with open(folder / "ResultManagement.json") as f:
            result_data = f.read()
            result_json = json.loads(result_data)
            result_timestamps = self._find_timestamps(result_json)
            if not result_timestamps:
                raise ValueError("No SourceTimestamp found in ResultManagement.json")
            original_timestamp = result_timestamps[0]
        
        # Load Trace.json
        with open(folder / "Trace.json") as f:
            trace_data = f.read()
            trace_json = json.loads(trace_data)
            trace_timestamps = self._find_timestamps(trace_json)
            if not trace_timestamps:
                raise ValueError("No SourceTimestamp found in Trace.json")
        
        # Load Heads.json
        with open(folder / "Heads.json") as f:
            heads_data = f.read()
            heads_json = json.loads(heads_data)
            heads_timestamps = self._find_timestamps(heads_json)
            if not heads_timestamps:
                raise ValueError("No SourceTimestamp found in Heads.json")
        
        # Cache the loaded data
        self._test_data_cache[folder_key] = (result_data, trace_data, heads_data, original_timestamp)
        
        return result_data, trace_data, heads_data, original_timestamp
    
    def _find_timestamps(self, data: Dict[str, Any]) -> List[str]:
        """Recursively find all SourceTimestamp values in a dictionary."""
        timestamps = []
        
        def _search(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "SourceTimestamp":
                        timestamps.append(value)
                    else:
                        _search(value)
            elif isinstance(obj, list):
                for item in obj:
                    _search(item)
        
        _search(data)
        return timestamps
    
    def inject_signal_tracking(self, data: str, signal_id: str) -> str:
        """Inject signal tracking ID into JSON data."""
        json_data = json.loads(data)
        json_data['_signal_id'] = signal_id
        return json.dumps(json_data)
    
    def log_signal(self, signal_id: str, toolbox_id: str, tool_id: str):
        """Log signal to CSV file."""
        with open(self.config.signal_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([signal_id, time.time(), toolbox_id, tool_id])
    
    def publish_message(self, topic: str, payload: str, original_timestamp: str, new_timestamp: str):
        """Publish a single message with timestamp replacement."""
        if not self.client:
            raise RuntimeError("MQTT client not connected. Call setup_client() first.")
            
        if original_timestamp in payload:
            payload = payload.replace(original_timestamp, new_timestamp)
        else:
            self.logger.warning(f"Original timestamp '{original_timestamp}' not found in payload")
        
        self.client.publish(topic, payload)
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.logger.debug(f"[{current_time}] Published to {topic} with timestamp {new_timestamp}")
        
        self._signal_counter += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get publishing statistics."""
        if not self._start_time:
            return {}
        
        elapsed = time.time() - self._start_time
        rate = self._signal_counter / elapsed if elapsed > 0 else 0
        
        return {
            'signals_sent': self._signal_counter,
            'elapsed_time': elapsed,
            'rate': rate,
            'average_interval': elapsed / self._signal_counter if self._signal_counter > 0 else 0
        }
    
    @abstractmethod
    def run(self):
        """Main publishing loop - to be implemented by subclasses."""
        pass