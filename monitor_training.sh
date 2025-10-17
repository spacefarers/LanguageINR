#!/bin/bash
# Monitor stage2 training progress

while true; do
    clear
    echo "Stage 2 Training Monitor"
    echo "========================"
    echo "Time: $(date '+%H:%M:%S')"
    echo

    # Check if process is running
    if ps aux | grep -q "[p]ython main.py --mode=stage2"; then
        echo "Status: RUNNING"
        ps aux | grep "[p]ython main.py --mode=stage2" | awk '{printf "CPU: %.1f%%  Memory: %.1fGB  Time: %s\n", $3, $6/1024/1024, $10}'
        echo
        echo "Waiting... (Ctrl+C to stop)"
    else
        echo "Status: COMPLETED or NOT RUNNING"
        [ -f "./models/stage2_semantic_head.pth" ] && echo "✓ Model saved!" || echo "✗ No model found"
        break
    fi
    sleep 5
done
