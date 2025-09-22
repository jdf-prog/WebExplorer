# How to inference or test?


## 📁 Project Structure

```
inference/
├── start_server.sh                   # Server startup script
├── run_webexplorer_inference.sh      # Main inference pipeline
├── run_multi_react.py                # Multi-turn reasoning engine
├── react_agent.py                    # ReAct agent implementation
├── tool_webexplorer_search.py        # Web search tool
├── tool_webexplorer_browse.py        # Web browsing tool
├── auto_judge.py                     # Automatic answer evaluation
└── eval_data/                        # Evaluation datasets
```


## ⚙️ Setup and Installation

**Quick Start**: Run `start_server.sh` to start VLLM servers, then `run_webexplorer_inference.sh` to run inference and evaluation.

### 1. Environment Variables

Create a configuration file or set the following environment variables:

```bash
# Model and Dataset Configuration
export MODEL_PATH=/path/to/your/webexplorer/model
export DATASET=browsecomp_validation_100
export OUTPUT_PATH=/path/to/output/directory

# Inference Parameters
export TEMPERATURE=0.6
export MAX_WORKERS=8
export JUDGE_ENGINE=deepseekchat
export ROLLOUT_COUNT=1

# API Keys for External Services
export SERPER_KEY_ID=your_serper_api_key_here
export JINA_API_KEYS=your_jina_api_key_here

# DeepSeek API Configuration (for judge engine and browse tool)
export DEEPSEEK_API_KEY=your_deepseek_api_key_here
export DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# Gemini API Configuration (alternative for browse tool)
export GEMINI_API_KEY=your_gemini_api_key_here
export GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta

# OpenAI API Configuration (optional, for summary model)
export API_KEY=your_openai_api_key_here
export API_BASE=https://api.openai.com/v1
export SUMMARY_MODEL_NAME=gpt-3.5-turbo
```

### 2. API Keys Setup

#### Serper API (Web Search)
- Set `SERPER_KEY_ID` environment variable

#### Jina API (Web Browsing)
- Set `JINA_API_KEYS` environment variable

#### LLMs used in Web Browsing selection
- **DeepSeek**: Get API key from [api.deepseek.com](https://api.deepseek.com)
- **Gemini**: Get API key from Google AI Studio
- **OpenAI**: Get API key from [platform.openai.com](https://platform.openai.com)

#### Judge Model
- Default DeepseekChat


## 🚀 Usage

### Step 1: Start VLLM Servers

First, start the VLLM inference servers:

```bash
chmod +x start_server.sh
./start_server.sh
```

This script will:
- Start 8 VLLM server instances on ports 6001-6008
- Wait for all servers to be ready
- Perform health checks on each server
- Create log files in `./vllm_logs/` directory


### Step 2: Run Inference and Evaluation

Once servers are ready, run the complete inference pipeline:

```bash
chmod +x run_webexplorer_inference.sh
./run_webexplorer_inference.sh
```



## 🐛 Troubleshooting

- **Server issues**: Check GPU memory and review logs in `./vllm_logs/`
- **API errors**: Verify all API keys are set correctly
- **Tool failures**: Check search and browse tools separately in `tool_webexplorer_browse.py` and `tool_webexplorer_search.py`

## 🙏 Acknowledgments

This generation framework is built upon the generation framework from [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch). We thank the authors for their excellent codebase and research contributions.