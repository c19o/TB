#!/bin/bash
# Overnight chain: wait for LSTM 15m → LSTM 5m → Exhaustive Optimizer
cd "C:/Users/C/Documents/Savage22 Server"

echo "=== OVERNIGHT RUN STARTED $(date) ==="

# Wait for current LSTM 15m to finish
echo "Waiting for LSTM 15m (already running)..."
while ps aux | grep -v grep | grep "lstm_sequence_model.*15m" > /dev/null 2>&1; do
    sleep 30
done
echo "LSTM 15m done at $(date)"

# Run LSTM 5m
echo "=== Starting LSTM 5m at $(date) ==="
python -u lstm_sequence_model.py --train --tf 5m 2>&1 | tee lstm_5m_log.txt
echo "LSTM 5m done at $(date)"

# Run exhaustive optimizer
echo "=== Starting Exhaustive Optimizer at $(date) ==="
python -u exhaustive_optimizer.py 2>&1 | tee optimizer_log.txt
echo "Optimizer done at $(date)"

echo "=== ALL DONE $(date) ==="
