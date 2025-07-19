#!/bin/sh
# Simple signal comparison for busybox

echo "=== Signal Tracking Report - $(date) ==="
echo

if [ -f /tracking/sent_signals.csv ]; then
    sent_count=$(wc -l /tracking/sent_signals.csv | cut -d' ' -f1)
    echo "Signals sent:     $sent_count"
else
    echo "Signals sent:     0"
fi

if [ -f /tracking/received_signals.csv ]; then
    received_count=$(wc -l /tracking/received_signals.csv | cut -d' ' -f1)
    echo "Messages received: $received_count"
    
    # Count unique signals
    unique_sent=$(cut -d, -f1 /tracking/sent_signals.csv | sort | uniq | wc -l)
    unique_received=$(cut -d, -f1 /tracking/received_signals.csv | sort | uniq | wc -l)
    echo "Unique signals sent:     $unique_sent"
    echo "Unique signals received: $unique_received"
    
    # Calculate missing
    missing=$((unique_sent - unique_received))
    echo
    echo "Missing signals: $missing"
else
    echo "Messages received: 0"
    echo
    echo "Missing signals: N/A (no received signals file)"
fi

echo "================================="