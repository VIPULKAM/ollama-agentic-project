# Observability Guide - Langfuse Integration

This document describes the planned Langfuse integration for the AI Coding Agent, providing tracing, monitoring, and analytics for LLM operations.

## Overview

[Langfuse](https://langfuse.com) is an open-source LLM observability platform that provides:
- **Tracing**: Capture all LLM calls, tool executions, and agent decisions
- **Analytics**: Token usage, latency metrics, cost tracking
- **Debugging**: Inspect prompts, responses, and error patterns
- **Sessions**: Link related traces across conversation turns

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Coding Agent                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Claude    │    │   Gemini    │    │   Ollama    │         │
│  │     LLM     │    │     LLM     │    │     LLM     │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                   ┌────────▼────────┐                           │
│                   │  LangGraph App  │                           │
│                   │  (ReAct Agent)  │                           │
│                   └────────┬────────┘                           │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                 │
│         │                  │                  │                 │
│  ┌──────▼──────┐   ┌───────▼───────┐  ┌───────▼───────┐        │
│  │  File Tools │   │   RAG Search  │  │   Git Tools   │        │
│  │  (6 tools)  │   │   (1 tool)    │  │   (5 tools)   │        │
│  └─────────────┘   └───────────────┘  └───────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Langfuse Callback Handler
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Langfuse Platform                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Traces    │  │  Analytics  │  │   Sessions  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# ===== LANGFUSE OBSERVABILITY =====
# Enable/disable Langfuse integration
ENABLE_LANGFUSE=false

# API Credentials (get from https://cloud.langfuse.com)
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key

# Host Configuration
# Options:
#   - https://cloud.langfuse.com     (EU region - default)
#   - https://us.cloud.langfuse.com  (US region)
#   - http://localhost:3000          (self-hosted)
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Sampling Rate (0.0 to 1.0)
# 1.0 = trace all requests (default)
# 0.1 = trace 10% of requests (for high-volume production)
LANGFUSE_SAMPLE_RATE=1.0

# Debug mode (enables SDK debug logging)
LANGFUSE_DEBUG=false
```

### Settings Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ENABLE_LANGFUSE` | bool | `false` | Master toggle for Langfuse integration |
| `LANGFUSE_PUBLIC_KEY` | str | `None` | Public API key from Langfuse dashboard |
| `LANGFUSE_SECRET_KEY` | str | `None` | Secret API key from Langfuse dashboard |
| `LANGFUSE_BASE_URL` | str | `cloud.langfuse.com` | Langfuse API endpoint |
| `LANGFUSE_SAMPLE_RATE` | float | `1.0` | Percentage of requests to trace (0.0-1.0) |
| `LANGFUSE_DEBUG` | bool | `false` | Enable SDK debug logging |

## What Gets Traced

### Automatic Tracing

When Langfuse is enabled, the following are automatically captured:

| Component | Captured Data |
|-----------|---------------|
| **LLM Calls** | Model name, input tokens, output tokens, latency, cost estimate, prompt, response |
| **Tool Executions** | Tool name, input parameters, output, duration, success/failure |
| **Agent Decisions** | ReAct thought process, action selections, observation handling |
| **Sessions** | All conversation turns linked by `session_id` |
| **Errors** | Exception type, message, stack trace, context |

### Trace Structure

Each `agent.ask()` call creates a trace with nested spans:

```
Trace (session_id: "default")
├── Generation: claude-sonnet-4 (prompt → initial response)
│   └── metadata: {tokens: 150, latency: 1.2s, cost: $0.0045}
├── Tool: read_file
│   └── input: {path: "/src/main.py"}
│   └── output: {content: "...", lines: 150}
│   └── duration: 0.05s
├── Tool: search_code
│   └── input: {pattern: "def main", path: "."}
│   └── output: {matches: 3}
│   └── duration: 0.12s
├── Generation: claude-sonnet-4 (with tool results)
│   └── metadata: {tokens: 280, latency: 1.8s, cost: $0.0084}
└── Final Response
```

### Metadata Tags

Traces are automatically tagged with:
- `provider:claude` / `provider:gemini` / `provider:ollama`
- `tools:true` / `tools:false`
- `session_id` for conversation linking

## Usage

### Enabling Langfuse

1. **Sign up** at [cloud.langfuse.com](https://cloud.langfuse.com)
2. **Create a project** and copy your API keys
3. **Configure** your `.env` file:
   ```bash
   ENABLE_LANGFUSE=true
   LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
   LANGFUSE_SECRET_KEY=sk-lf-xxxxx
   ```
4. **Run** the agent normally - traces appear automatically

### Viewing Traces

1. Go to your Langfuse dashboard
2. Navigate to **Traces** to see all captured interactions
3. Click on a trace to see:
   - Full prompt/response content
   - Token counts and costs
   - Tool execution details
   - Latency breakdown

### CLI Commands

Check Langfuse status:
```bash
python main.py
> info
# Shows: Langfuse: Enabled (or Disabled)
```

## Deployment Options

### Option 1: Langfuse Cloud (Recommended)

**Pros:**
- Zero infrastructure to manage
- Free tier: 50,000 observations/month
- Automatic updates and scaling
- SOC 2 Type II compliant

**Setup:**
```bash
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # EU
# or
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com  # US
```

### Option 2: Self-Hosted

**Requirements:**
- PostgreSQL (500MB+ RAM)
- ClickHouse (1-2GB RAM)
- Redis (optional, 100MB RAM)
- Docker/Docker Compose

**Setup:**
```bash
# Clone Langfuse
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Start with Docker Compose
docker compose up -d

# Configure agent
LANGFUSE_BASE_URL=http://localhost:3000
```

**When to self-host:**
- Data residency requirements
- Air-gapped environments
- Custom integrations needed
- High volume (>1M observations/month)

## Cost Tracking

Langfuse automatically estimates costs based on token usage. Cost rates are built-in for:

| Provider | Models | Cost Source |
|----------|--------|-------------|
| Anthropic | Claude 3/3.5/4 | Official pricing |
| Google | Gemini 1.5/2.5 | Official pricing |
| OpenAI | GPT-4/4o | Official pricing |
| Ollama | Local models | Free (no API costs) |

### Viewing Costs

In the Langfuse dashboard:
1. **Traces view**: See cost per interaction
2. **Analytics**: Aggregate costs by day/week/month
3. **Sessions**: Total cost per conversation

## Troubleshooting

### Langfuse Not Capturing Traces

1. **Check configuration:**
   ```bash
   python -c "from src.config.settings import settings; print(f'Enabled: {settings.ENABLE_LANGFUSE}, Keys: {bool(settings.LANGFUSE_PUBLIC_KEY)}')"
   ```

2. **Verify authentication:**
   ```python
   from langfuse import Langfuse
   lf = Langfuse()
   print(lf.auth_check())  # Should print True
   ```

3. **Enable debug mode:**
   ```bash
   LANGFUSE_DEBUG=true
   ```

### High Latency

If Langfuse adds noticeable latency:
- Traces are sent asynchronously (shouldn't block)
- Check network connectivity to Langfuse servers
- Consider reducing `LANGFUSE_SAMPLE_RATE` for production

### Missing Tool Traces

Tool executions should appear nested under the parent trace. If missing:
- Verify `ENABLE_TOOLS=true` in settings
- Check that callbacks are properly passed to `invoke()`

## Integration Points

### Code Locations

| File | Purpose |
|------|---------|
| `src/observability/__init__.py` | Module exports |
| `src/observability/langfuse_handler.py` | Callback handler singleton |
| `src/config/settings.py` | Configuration settings |
| `src/agent/agent.py:272-279` | Main integration in `ask()` |
| `src/agent/agent.py:398-406` | Streaming integration in `ask_stream()` |
| `src/cli/main.py` | Status display and shutdown |

### Adding Custom Metadata

To add custom metadata to traces:

```python
from src.observability import create_callback_config

config = create_callback_config(
    session_id="user-123-session",
    tags=["production", "high-priority"],
    metadata={
        "user_id": "user-123",
        "feature_flag": "new-rag-v2",
        "custom_field": "any-value"
    }
)
```

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [LangChain Integration Guide](https://langfuse.com/docs/integrations/langchain)
- [LangGraph Cookbook](https://langfuse.com/guides/cookbook/integration_langgraph)
- [Self-Hosting Guide](https://langfuse.com/docs/deployment/self-host)
- [Langfuse GitHub](https://github.com/langfuse/langfuse)

## Status

**Implementation Status:** Planned (not yet implemented)

**Plan File:** `/home/vipul/.claude/plans/spicy-imagining-hellman.md`

**Prerequisites:**
- Python 3.11+ (required for LangGraph integration)
- `langfuse>=3.0.0` package
- Langfuse account (cloud or self-hosted)
