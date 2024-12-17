import paho.mqtt.client as mqtt
import time
import json
import sysconfig
from pathlib import Path
import os

TOPIC = "test/topic"

client = mqtt.Client()
client.connect("localhost", 1883)

# abyss_path = Path(sysconfig.get_path('platlib')) / 'abyss'

# with open (abyss_path / 'examples-mqtt' /'data' / 'data-wh.json') as f:
#     d = f.read()

print(os.getcwd())
with open (f'{os.getcwd()}\\data\\Result.json') as f:
    d0 = f.read()
with open (f'{os.getcwd()}\\data\\Trace.json') as f:
    d1 = f.read()

def publish(payload, timestamp):
    # print(payload)
    d = json.loads(payload)
    # print(d['Messages'])
    d['Messages']['Timestamp'] = timestamp
    # client.publish("test/topic", json.dumps(payload))
    client.publish("TOPIC", json.dumps(d))
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"[{current_time}]: Published drilling MQTT data on {TOPIC} '{timestamp}'")

while True:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
    publish(d0, timestamp)
    publish(d1, timestamp)    
    time.sleep(5)