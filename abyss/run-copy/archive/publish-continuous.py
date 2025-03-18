import paho.mqtt.client as mqtt
import time
import json
import sysconfig
from pathlib import Path
import os

client = mqtt.Client()
client.connect("localhost", 1883)

# abyss_path = Path(sysconfig.get_path('platlib')) / 'abyss'

# with open (abyss_path / 'examples-mqtt' /'data' / 'data-wh.json') as f:
#     d = f.read()

print(os.getcwd())
with open (f'{os.getcwd()}\\data\\data-wh.json') as f:
    d = f.read()

while True:
    payload = d
    client.publish("test/topic", json.dumps(payload))
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f"[{current_time}]: Published drilling MQTT data on test/topic")
    time.sleep(5)