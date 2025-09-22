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

##############hyperparams################
export MODEL_PATH=/path/to/your/webexplorer/model
export DATASET=your_dataset_name
export OUTPUT_PATH=/path/to/output/directory
export ROLLOUT_COUNT=1 # eval avg@3
export TEMPERATURE=0.6
export MAX_WORKERS=8  # Optimized: 2x number of vLLM servers (8 servers * 2)

export JUDGE_ENGINE=deepseekchat


## API Keys for external services
export SERPER_KEY_ID=your_serper_key_here
export JINA_API_KEYS=your_jina_key_here


## DeepSeek API configuration (for judge engine)
export DEEPSEEK_API_KEY=your_deepseek_key_here
export DEEPSEEK_API_BASE=https://api.deepseek.com/v1


# For browse tool, you can choose gemini or deepseekchat or openai, default is deepseekchat

## Gemini API configuration
export GEMINI_API_KEY=your_gemini_key_here
export GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta

## OpenAI API configuration (optional, for summary model)
export API_KEY=your_openai_key_here
export API_BASE=https://api.openai.com/v1
export SUMMARY_MODEL_NAME=gpt-3.5-turbo


export TORCH_COMPILE_CACHE_DIR="./cache"

######################################
### 1. start server           ###
######################################

# echo "Starting VLLM servers..."
# CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6001 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6002 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6003 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6004 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6005 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6006 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6007 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6008 --disable-log-requests &

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

#####################################
### 3. start infer               ####
#####################################

echo "==== start WebExplorer evaluation... ===="

cd "$( dirname -- "${BASH_SOURCE[0]}" )"

# Enable judge engine
python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT --auto_judge --judge_engine $JUDGE_ENGINE
