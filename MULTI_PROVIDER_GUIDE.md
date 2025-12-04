# Multi-Provider Setup Guide (Version 2.0)

**AI Coding Agent now supports both Ollama (self-hosted) and Claude API (Anthropic)!**

---

## What's New in Version 2.0?

✅ **Three Operating Modes:**
1. **Ollama Mode** - Use self-hosted models (local or cloud)
2. **Claude Mode** - Use Anthropic's Claude API (Haiku for cost-effectiveness)
3. **Hybrid Mode** - Smart routing between both providers

✅ **Zero Code Changes** - Just update .env configuration!

✅ **Backward Compatible** - Existing Ollama-only setup still works

---

## Quick Start

### Option 1: Continue with Ollama Only (No Changes Needed)

Your existing setup works perfectly! No action required.

```bash
# .env (current configuration - no changes needed)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=codellama:7b
```

---

### Option 2: Use Claude API Only

**Step 1: Get Anthropic API Key**
- Go to: https://console.anthropic.com/
- Sign up or log in
- Navigate to API Keys
- Create a new API key
- Copy the key (starts with `sk-ant-...`)

**Step 2: Update .env**
```bash
# .env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
CLAUDE_MODEL=claude-3-haiku-20240307
```

**Step 3: Test**
```bash
python main.py
# Ask: "Write a Python function to validate emails"
```

**Cost:** ~$0.25 per 1M input tokens, $1.25 per 1M output tokens

---

### Option 3: Hybrid Mode (RECOMMENDED!)

**Best of both worlds:** Use Ollama for routine queries, Claude for complex tasks.

**Step 1: Configure .env**
```bash
# .env
LLM_PROVIDER=hybrid

# Ollama setup (for routine queries)
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=codellama:7b

# Claude setup (for complex queries)
ANTHROPIC_API_KEY=sk-ant-your-key-here
CLAUDE_MODEL=claude-3-haiku-20240307
```

**Step 2: Test Routing**

**Simple query (routes to Ollama):**
```
You: Write a SQL query to count users
[Hybrid Mode: Using Ollama]
Agent: SELECT COUNT(*) FROM users;
```

**Complex query (routes to Claude):**
```
You: Review my architecture design
[Hybrid Mode: Using Claude for complex query]
Agent: <detailed architectural review>
```

**Keywords that trigger Claude:**
- architecture
- design pattern
- refactor
- optimize
- security
- best practice
- review
- compare

**Step 3: Manual Override**
You can force a specific provider for any query:
```python
# Use Ollama for this query
response = agent.ask("Write Python code", force_provider="ollama")

# Use Claude for this query
response = agent.ask("Simple query", force_provider="claude")
```

---

## Cost Comparison

### Example: 180,000 queries/month (400 users × 15 queries/day)

| Mode | Monthly Cost | Per User | Notes |
|------|--------------|----------|-------|
| **Ollama Only (CPU)** | $100 | $0.25 | Fixed cost, slower (10-20s) |
| **Ollama Only (GPU)** | $520-730 | $1.30-1.83 | Fixed cost, fast (2-5s) |
| **Claude Haiku Only** | $77 | $0.19 | Scales with usage |
| **Claude Sonnet Only** | $918 | $2.30 | Best quality, expensive |
| **Hybrid** (80% Ollama, 20% Claude) | $540-615 | $1.35-1.54 | Best value! |

**Hybrid Mode Savings:**
- vs Pure Claude Sonnet: 33-67% cheaper
- vs Pure Ollama GPU: Similar cost, better quality for complex queries

---

## Configuration Reference

### Environment Variables

```bash
# Required: Choose your provider
LLM_PROVIDER=ollama|claude|hybrid

# Ollama Configuration (required if LLM_PROVIDER="ollama" or "hybrid")
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=codellama:7b

# Claude Configuration (required if LLM_PROVIDER="claude" or "hybrid")
ANTHROPIC_API_KEY=sk-ant-your-key-here
CLAUDE_MODEL=claude-3-haiku-20240307

# Alternative Claude Models:
# - claude-3-haiku-20240307 (cheapest, fast)
# - claude-3-5-sonnet-20241022 (best quality, expensive)

# Model Parameters
TEMPERATURE=0.1
MAX_TOKENS=2000

# Memory
MAX_HISTORY_LENGTH=10
```

---

## Usage Examples

### Python API

```python
from src.agent.agent import CodingAgent

# Ollama only
agent = CodingAgent(provider="ollama")
response = agent.ask("Write a FastAPI endpoint")

# Claude only
agent = CodingAgent(provider="claude", api_key="sk-ant-...")
response = agent.ask("Review my code architecture")

# Hybrid mode
agent = CodingAgent(provider="hybrid")
response = agent.ask("Write SQL query")  # → Ollama
response = agent.ask("Optimize my design")  # → Claude

# Force specific provider
response = agent.ask("Write code", force_provider="claude")
```

### CLI

```bash
# Update .env with desired configuration
LLM_PROVIDER=hybrid
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Run agent
python main.py

# Queries automatically route based on complexity!
```

---

## Troubleshooting

### Error: "ANTHROPIC_API_KEY must be set"

**Solution:**
```bash
# Add to .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Error: "Make sure Ollama is running"

**Solution:**
```bash
# Start Ollama
ollama serve

# Or use Claude mode if you don't want to run Ollama
LLM_PROVIDER=claude
```

### Hybrid Mode Always Using Ollama

**Solution:** Check if your query contains routing keywords
```bash
# These trigger Claude:
"architecture review"
"optimize design"
"security best practices"

# These use Ollama:
"write sql query"
"create function"
"simple code"
```

### Claude API Rate Limits

**Error:** `RateLimitError: rate_limit_exceeded`

**Solution:**
```bash
# Switch to Ollama temporarily
LLM_PROVIDER=ollama

# Or upgrade your Claude API tier at:
# https://console.anthropic.com/settings/limits
```

---

## Migration Guide

### From Version 1.0 (Ollama Only)

**Good news:** Your existing setup works without changes!

**Optional Upgrade:**
```bash
# 1. Update dependencies
uv pip install langchain-anthropic anthropic

# 2. Get Claude API key (optional)
# https://console.anthropic.com/

# 3. Update .env (optional)
LLM_PROVIDER=hybrid  # or keep "ollama"
ANTHROPIC_API_KEY=sk-ant-your-key-here  # if using hybrid/claude

# 4. Test
python main.py
```

---

## Performance Comparison

| Provider | Response Time | Quality | Cost (per query) |
|----------|---------------|---------|------------------|
| **Ollama (CPU)** | 10-20s | Good | $0 |
| **Ollama (GPU)** | 2-5s | Good | $0 |
| **Claude Haiku** | 2-3s | Very Good | ~$0.0005 |
| **Claude Sonnet** | 3-5s | Excellent | ~$0.005 |

**Hybrid Mode:** 80-90% queries use Ollama (fast, free), 10-20% use Claude (best quality)

---

## Best Practices

### 1. Start with Hybrid Mode
```bash
LLM_PROVIDER=hybrid
```
- Most cost-effective
- Best quality for complex queries
- Fast for routine queries

### 2. Monitor Usage
```bash
# Check what provider is being used
# Output shows: [Hybrid Mode: Using Ollama] or [Hybrid Mode: Using Claude]
```

### 3. Adjust Keywords for Your Team
```bash
# In .env or settings.py
CLAUDE_KEYWORDS=architecture,refactor,security,optimize,review,design
```

### 4. Force Provider When Needed
```python
# For critical architectural decisions
response = agent.ask("Design system architecture", force_provider="claude")

# For simple, frequent queries
response = agent.ask("Write CRUD endpoint", force_provider="ollama")
```

---

## Roadmap

### Version 2.1 (Planned)
- [ ] Automatic cost tracking
- [ ] Usage analytics dashboard
- [ ] More intelligent routing (ML-based)
- [ ] Support for more providers (OpenAI GPT-4, etc.)

### Version 2.2 (Planned)
- [ ] Multi-provider load balancing
- [ ] Fallback chains (Ollama → Claude → GPT-4)
- [ ] Custom routing rules per user

---

## FAQ

**Q: Can I use both Ollama and Claude simultaneously?**
A: Yes! Use `LLM_PROVIDER=hybrid` mode.

**Q: How much does Claude Haiku cost?**
A: ~$0.26/user/month for 400 users doing 15 queries/day.

**Q: Which is better: Ollama or Claude?**
A: Hybrid mode! Ollama for speed/cost, Claude for quality when needed.

**Q: Can I change providers without restarting?**
A: Not yet. Restart the agent after updating .env. (Feature coming in v2.1)

**Q: Does this work with cloud-hosted Ollama?**
A: Yes! Just set `OLLAMA_BASE_URL=https://your-server.com`

**Q: Can I use Claude Sonnet instead of Haiku?**
A: Yes! Set `CLAUDE_MODEL=claude-3-5-sonnet-20241022` (5x more expensive)

---

## Support

**Issues?** Check troubleshooting section above or:
- Review .env configuration
- Check API key is valid
- Verify Ollama is running (if applicable)
- Check network connectivity

**Feature Requests?** Open an issue in the project repository.

---

**Version 2.0 gives you flexibility:** Start with pure Ollama, add Claude when budget allows, or use hybrid for best results!
