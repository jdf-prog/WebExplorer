#!/bin/bash

export TORCHDYNAMO_VERBOSE=1
export TORCHDYNAMO_DISABLE=1
export NCCL_IB_TC=16
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
export GLOO_SOCKET_IFNAME=eth0
export PYTHONDONTWRITEBYTECODE=1

##############server config################
export MODEL_PATH=/data/minimax-dialogue/junteng/HF_HOME/WebExplorer

######################################
### 1. start server           ###
######################################

echo "Starting VLLM servers..."
# 创建日志目录
mkdir -p ./vllm_logs

echo "Server logs will be saved to ./vllm_logs/"
echo "Using --gpu-memory-utilization 0.8 for detailed logging"

CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6001 --gpu-memory-utilization 0.8 > ./vllm_logs/server_6001.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6002 --gpu-memory-utilization 0.8 > ./vllm_logs/server_6002.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6003 --gpu-memory-utilization 0.8 > ./vllm_logs/server_6003.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6004 --gpu-memory-utilization 0.8 > ./vllm_logs/server_6004.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6005 --gpu-memory-utilization 0.8 > ./vllm_logs/server_6005.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6006 --gpu-memory-utilization 0.8 > ./vllm_logs/server_6006.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6007 --gpu-memory-utilization 0.8 > ./vllm_logs/server_6007.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6008 --gpu-memory-utilization 0.8 > ./vllm_logs/server_6008.log 2>&1 &

#######################################################
### 2. Waiting for the server port to be ready  ###
#######################################################

timeout=6000
start_time=$(date +%s)

main_ports=(6001 6002 6003 6004 6005 6006 6007 6008)
echo "Mode: All ports used as main model"

declare -A server_status
for port in "${main_ports[@]}"; do
    server_status[$port]=false
done

echo "Waiting for servers to start..."

while true; do
    all_ready=true
    
    for port in "${main_ports[@]}"; do
        if [ "${server_status[$port]}" = "false" ]; then
            if curl -s -f http://localhost:$port/v1/models > /dev/null 2>&1; then
                echo "Main model server (port $port) is ready!"
                server_status[$port]=true
            else
                all_ready=false
            fi
        fi
    done
    
    if [ "$all_ready" = "true" ]; then
        echo "All servers are ready for inference!"
        break
    fi
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\nError: Server startup timeout after ${timeout} seconds"
        
        for port in "${main_ports[@]}"; do
            if [ "${server_status[$port]}" = "false" ]; then
                echo "Main model server (port $port) failed to start"
            fi
        done

        
        exit 1
    fi
    
    printf 'Waiting for servers to start .....'
    sleep 10
done

failed_servers=()
for port in "${main_ports[@]}"; do
    if [ "${server_status[$port]}" = "false" ]; then
        failed_servers+=($port)
    fi
done

if [ ${#failed_servers[@]} -gt 0 ]; then
    echo "Error: The following servers failed to start: ${failed_servers[*]}"
    exit 1
else
    echo "All required servers are running successfully!"
fi
