# AI Coding Agent

**Local MVP** with **AWS Cloud Migration** support

An intelligent coding assistant powered by LangChain and Ollama, specialized in Python, TypeScript, and database technologies.

---

## Features

- ✅ **Multi-language Support**: Python, TypeScript/JavaScript
- ✅ **Database Expertise**: PostgreSQL, MySQL, MSSQL, MongoDB, Redis, Snowflake, ClickHouse
- ✅ **LangChain Integration**: Production-ready LLM orchestration
- ✅ **Conversation Memory**: Maintains context across questions
- ✅ **Beautiful CLI**: Rich terminal interface with syntax highlighting
- ✅ **Cloud-Ready**: Easy migration to AWS (just change one URL!)

---

## Quick Start

### Prerequisites

1. **Python 3.10+**
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Ollama** (for local inference)
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ```

3. **CodeLlama Model**
   ```bash
   ollama pull codellama:7b
   ```

### Installation

1. **Clone/Navigate to project**
   ```bash
   cd ollama-agentic-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work for local)
   ```

### Run the Agent

```bash
python main.py
```

That's it! The agent is now running locally.

---

## Usage

### Interactive CLI

```
You: Write a PostgreSQL query to find duplicate emails

Agent: Here's a query to find duplicate emails:

SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1
ORDER BY count DESC;
```

### Commands

- **Type your question** - Ask any coding question
- **`clear`** - Clear conversation history
- **`info`** - Show model configuration
- **`help`** - Show help message
- **`exit`** - Exit the application

---

## Example Questions

Try asking:

- "Write a Python FastAPI endpoint for user authentication"
- "Create a MongoDB aggregation pipeline for sales analytics"
- "Show me how to optimize a slow PostgreSQL query"
- "Write a TypeScript function to validate email addresses"
- "What's the best way to handle errors in async Python?"

---

## Cloud Migration (AWS)

### Why Migrate to Cloud?

**Local Development Issues:**
- Laptop gets hot during inference
- 35+ second response times
- Can't use laptop for other tasks
- Inconsistent across team

**Cloud Benefits:**
- ✅ Faster responses (better CPU/GPU)
- ✅ No laptop resource usage
- ✅ Shared by entire team
- ✅ Always available
- ✅ Cost: ~$2-4/developer/month (CPU) or $4-15/developer/month (GPU)

### Migration Steps

**Deployment Options:**

| Option | Cost/Month | Response Time | Best For |
|--------|------------|---------------|----------|
| **CPU (c7g.xlarge)** | $100 | 10-20s | Initial rollout, budget-conscious |
| **GPU Spot (g5.xlarge)** | $220 | 2-5s ⚡ | Production, 30+ developers |
| **GPU On-Demand** | $730 | 2-5s ⚡ | Mission-critical, maximum uptime |

**Step 1: Provision AWS Server**

```bash
# Option A: CPU (Start here)
# Recommended: c7g.xlarge (ARM, 4 vCPU, 8GB RAM)
# Cost: ~$100/month

# Option B: GPU (Upgrade later for 10x speed)
# Recommended: g5.xlarge spot (NVIDIA A10G, 24GB VRAM)
# Cost: ~$220/month
# See GPU_DEPLOYMENT_GUIDE.md for details
```

**Step 2: Install Ollama on Server**

```bash
# SSH into your server
ssh ubuntu@your-server.com

# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull codellama:7b

# Run as service
ollama serve
```

**Step 3: Update .env File**

```bash
# Change from:
OLLAMA_BASE_URL=http://localhost:11434

# To:
OLLAMA_BASE_URL=https://ai-server.yourcompany.com
```

**That's it!** The code works identically.

### Cloud Deployment Architecture

```
Developer Laptops (CLI/Browser)
        ↓ HTTPS
AWS EC2 Server
  ├─ FastAPI (future)
  └─ Ollama + CodeLlama
```

See `DEPLOYMENT_ANALYSIS.md` for detailed cloud deployment guide.

---

## Project Structure

```
ollama-agentic-project/
├── src/
│   ├── agent/
│   │   ├── agent.py          # LangChain agent (20 lines!)
│   │   └── prompts.py        # Database knowledge
│   ├── cli/
│   │   └── main.py           # Terminal interface
│   └── config/
│       └── settings.py       # Configuration
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT_ANALYSIS.md
│   └── EXECUTIVE_DECISION_BRIEF.md
├── main.py                   # Entry point
├── requirements.txt
└── .env                      # Configuration
```

---

## Configuration

Edit `.env` to customize:

```bash
# Model Configuration
OLLAMA_BASE_URL=http://localhost:11434  # Change for cloud
MODEL_NAME=codellama:7b                 # Or llama3.1:8b

# Model Parameters
TEMPERATURE=0.1                         # Lower = more focused
MAX_TOKENS=2000

# Memory
MAX_HISTORY_LENGTH=10                   # Conversation turns
```

---

## Development

### Install dev dependencies

```bash
pip install -r requirements-dev.txt
```

### Run tests

```bash
pytest
```

### Code formatting

```bash
black src/
ruff check src/
```

### Type checking

```bash
mypy src/
```

---

## Troubleshooting

### "Ollama connection error"

**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list
```

### "Model not found"

**Solution:**
```bash
# Pull the model
ollama pull codellama:7b
```

### "Slow responses (>60 seconds)"

**Reasons:**
- Older CPU (expected on 2013-era hardware)
- Model loading for first query

**Solutions:**
- Use smaller model: `llama3.2:3b`
- Migrate to cloud (faster CPU)

### "Laptop getting hot"

**This is expected!** Local inference is CPU-intensive.

**Solutions:**
- Use for development/testing only
- Deploy to cloud for team use
- See `DEPLOYMENT_ANALYSIS.md`

---

## Cost Analysis

### Local (Current)
- **Cost:** $0
- **Performance:** Slow (35s), laptop gets hot
- **Best for:** Development, testing, MVP demo

### Cloud (Recommended for Production)
- **Cost:** $35-150/month ($0.70-3/developer)
- **Performance:** Fast (10-15s), no laptop impact
- **Best for:** Team deployment, daily use

See `EXECUTIVE_DECISION_BRIEF.md` for ROI analysis.

---

## Technical Stack

- **LangChain**: LLM orchestration
- **Ollama**: Local LLM inference
- **CodeLlama:7b**: Code-specialized model (Meta)
- **Pydantic**: Configuration management
- **Rich**: Beautiful terminal UI

---

## Documentation

- **ARCHITECTURE.md** - Technical architecture (LangChain-based)
- **DEPLOYMENT_ANALYSIS.md** - Cloud deployment options (CPU & GPU)
- **GPU_DEPLOYMENT_GUIDE.md** - GPU deployment strategy & ROI analysis
- **DESIGN_SUMMARY.md** - Quick reference guide
- **EXECUTIVE_DECISION_BRIEF.md** - For management approval
- **MODEL_COMPARISON_REPORT.md** - Model selection rationale

---

## Roadmap

### Phase 1: Local MVP ✅
- [x] LangChain integration
- [x] Database knowledge prompts
- [x] CLI interface
- [x] Conversation memory
- [ ] Testing with team

### Phase 2: Cloud Deployment
- [ ] AWS EC2 provisioning
- [ ] FastAPI backend
- [ ] Web UI
- [ ] Authentication
- [ ] Team rollout

### Phase 3: Advanced Features
- [ ] RAG with company documentation
- [ ] Code execution sandbox
- [ ] IDE plugins (VS Code)
- [ ] Analytics dashboard

---

## Contributing

This is an internal tool. For improvements:

1. Test locally
2. Update documentation
3. Share with team
4. Iterate based on feedback

---

## License

Internal use only.

---

## Support

**Issues?** Contact the engineering team.

**Feature requests?** Open a discussion.

**Questions?** Check the docs/ folder.

---

## Acknowledgments

- **LangChain** - LLM orchestration framework
- **Ollama** - Local LLM inference
- **Meta** - CodeLlama model
- **Anthropic** - Claude (for this README! 😄)
