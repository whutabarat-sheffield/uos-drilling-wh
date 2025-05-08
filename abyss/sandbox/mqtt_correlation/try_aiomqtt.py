import asyncio
import aiomqtt
from collections import UserDict
from typing import Any, Callable, Coroutine, Optional, Union
from aiomqtt import Client, Message, Topic

class DataBuffer

async def listener(client):
    await client.subscribe("temperature/#")
    await client.subscribe("humidity/#")
    async for message in client.messages:
        if message.topic.matches("humidity/inside"):
            print(f"[humidity/inside] {message.payload}")
        if message.topic.matches("+/outside"):
            print(f"[+/outside] {message.payload}")
        if message.topic.matches("temperature/#"):
            print(f"[temperature/#] {message.payload}")

async def publisher(client):

    await client.publish("temperature/inside", "22.5")
    await client.publish("humidity/outside", "45")
    await client.publish("humidity/inside", "50")
    await client.publish("temperature/outside", "25")

async def main():
    async with aiomqtt.Client("OPCPUBSUB") as client:
        await publisher(client)
        await listener(client)


asyncio.run(main())