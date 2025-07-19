#!/bin/bash
# Signal comparison for local execution (not in container)

TRACKING_DIR="${TRACKING_DIR:-tracking}"

echo "=== Signal Tracking Report - $(date) ==="
echo
echo "Signals sent:     $(wc -l < $TRACKING_DIR/sent_signals.csv 2>/dev/null || echo 0)"
echo "Messages received: $(wc -l < $TRACKING_DIR/received_signals.csv 2>/dev/null || echo 0)"

# Count unique signal IDs
sent_unique=$(cut -d, -f1 $TRACKING_DIR/sent_signals.csv 2>/dev/null | sort | uniq | wc -l)
received_unique=$(cut -d, -f1 $TRACKING_DIR/received_signals.csv 2>/dev/null | sort | uniq | wc -l)

echo "Unique signals sent:     $sent_unique"
echo "Unique signals received: $received_unique"

# Show missing signals (sent but not received)
echo
echo "Missing signals:"
comm -23 <(cut -d, -f1 $TRACKING_DIR/sent_signals.csv 2>/dev/null | sort | uniq) \
         <(cut -d, -f1 $TRACKING_DIR/received_signals.csv 2>/dev/null | sort | uniq) | wc -l

# Calculate latency if both files exist
if [ -f $TRACKING_DIR/sent_signals.csv ] && [ -f $TRACKING_DIR/received_signals.csv ]; then
    echo
    echo "Latency statistics (first receipt):"
    # For each signal, get first sent time and first received time
    join -t, -1 1 -2 1 \
        <(cut -d, -f1,2 $TRACKING_DIR/sent_signals.csv | sort -t, -k1,1 | sort -u -t, -k1,1) \
        <(cut -d, -f1,2 $TRACKING_DIR/received_signals.csv | sort -t, -k1,1 | sort -u -t, -k1,1) | \
    awk -F, '{latency=($3-$2)*1000; print latency}' | \
    awk '{
        sum+=$1; 
        if(NR==1){min=max=$1}; 
        if($1>max){max=$1}; 
        if($1<min){min=$1}
    } 
    END {
        if(NR>0) {
            printf "  Average: %.1f ms\n", sum/NR
            printf "  Min:     %.1f ms\n", min
            printf "  Max:     %.1f ms\n", max
        }
    }'
fi

echo "================================="