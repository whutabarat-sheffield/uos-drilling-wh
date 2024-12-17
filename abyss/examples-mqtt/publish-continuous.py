import paho.mqtt.client as mqtt
import time
import json
# import sysconfig
# from pathlib import Path
import os
# import yaml

TOPIC_RESULT = "ResultManagement/Result"
TOPIC_TRACE = "ResultManagement/Trace"

client = mqtt.Client()
client.connect("localhost", 1883)

# abyss_path = Path(sysconfig.get_path('platlib')) / 'abyss'

# with open (abyss_path / 'examples-mqtt' /'data' / 'data-wh.json') as f:
#     d = f.read()

print(os.getcwd())
with open (f'{os.getcwd()}\\data\\Result.json') as f:
    d_result = f.read()
with open (f'{os.getcwd()}\\data\\Trace.json') as f:
    d_trace = f.read()

def publish(topic, payload, timestamp):
    # print(payload)
    d = json.loads(payload)
    # print(d['Messages'])
    d['Messages']['Timestamp'] = timestamp
    # client.publish("test/topic", json.dumps(payload))
    client.publish(topic, json.dumps(d))
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"[{current_time}]: Publish data on {topic} '{timestamp}'")

while True:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
    publish(TOPIC_RESULT, d_result, timestamp)
    publish(TOPIC_TRACE, d_trace, timestamp)    
    time.sleep(5)