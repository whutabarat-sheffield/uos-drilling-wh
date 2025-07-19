#!/bin/bash
# Simple script to start the signal monitor with paho-mqtt installed

echo "Installing paho-mqtt..."
pip install paho-mqtt >/dev/null 2>&1

echo "Starting signal monitor..."
python /app/signal_monitor.py