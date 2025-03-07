# Find the process ID using the port
lsof -i :13141

# Kill the process (replace PID with the process ID from above command)
kill -9 $(lsof -t -i:13141)

# Set CUDA device before running vllm serve
export CUDA_VISIBLE_DEVICES=4

MODEL_PATH=/shared/eng/zifengw2/mistral_models/leads-mistral-7b-v0.3
vllm serve $MODEL_PATH --config vllm_config.yaml > vllm.log 2>&1 &
