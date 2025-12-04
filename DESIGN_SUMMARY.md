# AI Coding Agent - Design Summary (LangChain-Based)

**Quick Reference Guide - Version 2.0**

---

## High-Level Architecture

```
┌─────────────┐
│    User     │ (Types coding question)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  CLI App    │ (Beautiful terminal with Rich)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Agent Wrapper│ (Thin wrapper - 20 lines!)
│ (Our Code)  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│   LangChain ConversationChain    │
│  ┌────┐  ┌──────┐  ┌─────────┐  │
│  │LLM │  │Memory│  │ Prompts │  │
│  └────┘  └──────┘  └─────────┘  │
└──────────────┬───────────────────┘
               │
               ▼
        ┌─────────────┐
        │  CodeLlama  │
        │    :7b      │
        └─────────────┘
```

**Key Win:** LangChain does 80% of the work!

---

## What Changed from v1.0?

| Aspect | v1.0 (Custom) | v2.0 (LangChain) | Benefit |
|--------|---------------|------------------|---------|
| **Code to write** | ~500 lines | ~100 lines | 80% less |
| **Components** | 9 files | 5 files | Simpler |
| **Dependencies** | 5 packages | 6 packages (+LangChain) | Worth it |
| **Time to MVP** | 2-3 weeks | 3-5 days | 5x faster |
| **Maintenance** | High | Low | Less burden |
| **Future features** | Build from scratch | Built-in | Free upgrades |

---

## Key Components

### 1. **Agent Wrapper** (`agent.py`) - 20 lines!
```python
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

class CodingAgent:
    def __init__(self):
        self.llm = Ollama(model="codellama:7b", temperature=0.1)
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(llm=self.llm, memory=self.memory)

    def ask(self, query: str) -> str:
        return self.chain.run(query)
```

**That's it!** LangChain handles everything else.

### 2. **Custom Prompts** (`prompts.py`)
- Database expertise (PostgreSQL, MySQL, MongoDB, etc.)
- Python & TypeScript best practices
- LangChain PromptTemplate integration

### 3. **CLI** (`main.py`)
- Rich library for beautiful output
- Interactive REPL
- Markdown rendering

### 4. **Config** (`settings.py`)
- Pydantic for type-safe settings
- .env file support

---

## Technology Stack

### Core Dependencies
```
langchain              # Main framework
langchain-community    # Ollama integration
pydantic              # Config validation
pydantic-settings     # Settings management
rich                  # Terminal UI
python-dotenv         # Environment variables
```

### Why LangChain?

✅ **Built-in Features**
- Ollama integration
- Memory management
- Prompt templating
- Streaming
- Error handling

✅ **Future Ready**
- RAG (vector stores)
- Tools & agents
- Multi-agent orchestration
- LangServe API

✅ **Production Ready**
- Battle-tested
- Security updates
- Active community
- Great docs

---

## File Structure (Simplified!)

```
ollama-agentic-project/
├── src/
│   ├── agent/
│   │   ├── agent.py      ← 20 lines with LangChain!
│   │   └── prompts.py    ← Custom database knowledge
│   ├── cli/
│   │   └── main.py       ← Terminal interface
│   └── config/
│       └── settings.py   ← Configuration
├── docs/
│   ├── ARCHITECTURE.md
│   └── DESIGN_SUMMARY.md
├── requirements.txt      ← Simple dependencies
├── .env.example
├── main.py              ← Entry point
└── README.md
```

**9 files → 5 core files**

---

## Data Flow Example

**User:** "Write a PostgreSQL query to find duplicates"

```
1. CLI captures input
   ↓
2. agent.ask(query)
   ↓
3. LangChain ConversationChain:
   • Retrieves conversation history (automatic)
   • Formats prompt with template (automatic)
   • Calls Ollama LLM (automatic)
   ↓
4. CodeLlama generates SQL
   ↓
5. LangChain:
   • Stores in memory (automatic)
   • Returns response
   ↓
6. CLI displays with Rich formatting
```

**Steps 3-5 are automatic with LangChain!**

---

## Configuration (.env)

```bash
# Model
MODEL_NAME=codellama:7b
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0.1
MAX_TOKENS=2000

# Memory
MAX_HISTORY_LENGTH=10
MEMORY_TYPE=buffer

# CLI
THEME=monokai
STREAM_OUTPUT=true
```

---

## Design Decisions

### ✅ Use LangChain (UPDATED)
**Why:**
- Industry standard
- 80% less code
- Battle-tested
- Future-proof

**Trade-off:** +100MB dependencies
**Verdict:** Massive win

### ✅ CodeLlama:7b
**Why:**
- Fast (35s response)
- Code-specialized
- US-based (Meta)
- Free

### ✅ CLI First
**Why:**
- Developers prefer terminal
- Faster to build
- Can add Web UI later

### ✅ ConversationBufferMemory
**Why:**
- Simplest for MVP
- Full context
- Can upgrade later

---

## MVP Features (Phase 1)

**With LangChain, these are trivial:**

- [x] Model selection & testing
- [ ] LangChain integration (simple!)
- [ ] Custom prompts (database knowledge)
- [ ] Conversation memory (built-in)
- [ ] Interactive CLI (Rich)
- [ ] Code syntax highlighting
- [ ] Error handling (built-in)

**Time estimate:** 3-5 days vs 2-3 weeks!

---

## Future Features (Easy with LangChain!)

### Phase 2
```python
# RAG - 5 lines with LangChain!
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

vectorstore = Chroma.from_documents(company_docs)
qa = RetrievalQA.from_chain_type(llm=ollama, retriever=vectorstore)
```

### Phase 3
```python
# Tools - Use LangChain agents
from langchain.agents import initialize_agent, Tool

tools = [
    Tool(name="Calculator", func=calculator),
    Tool(name="WebSearch", func=search)
]
agent = initialize_agent(tools, llm)
```

### Phase 4
```python
# Production API - LangServe
from langserve import add_routes

app = FastAPI()
add_routes(app, chain, path="/chat")
# Done! Production API ready.
```

---

## How to Use (Preview)

```bash
$ python main.py

🤖 AI Coding Agent v2.0 (Powered by LangChain)
Model: CodeLlama:7b
Type 'help' for commands, 'exit' to quit

You: How do I create a MongoDB index?

Agent: To create an index in MongoDB:

```javascript
db.collection.createIndex({ fieldName: 1 })
```

For compound index:
```javascript
db.users.createIndex({ email: 1, created_at: -1 })
```

1 = ascending, -1 = descending

You: What about unique indexes?

Agent: Add unique: true option:

```javascript
db.users.createIndex(
  { email: 1 },
  { unique: true }
)
```

This ensures no duplicate emails.
```

---

## LangChain Benefits Summary

| Feature | Custom Code | LangChain | Winner |
|---------|-------------|-----------|--------|
| Ollama integration | 100 lines | 1 line | 🏆 LangChain |
| Memory management | 80 lines | 1 line | 🏆 LangChain |
| Prompt templates | 50 lines | 5 lines | 🏆 LangChain |
| Streaming | 60 lines | Built-in | 🏆 LangChain |
| Error handling | 40 lines | Built-in | 🏆 LangChain |
| RAG (future) | 200+ lines | 10 lines | 🏆 LangChain |
| Agents (future) | 300+ lines | 15 lines | 🏆 LangChain |

**Total savings: ~800 lines of code!**

---

## Success Metrics

| Metric | Target | Expected |
|--------|--------|----------|
| Development time | < 2 weeks | 3-5 days ✅ |
| Lines of code | < 200 | ~100 ✅ |
| Response time | < 60s | 35s ✅ |
| Code accuracy | > 90% | TBD |
| Team adoption | > 50% | TBD |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LangChain too complex | Excellent docs, large community |
| Model not good enough | Easy to swap (1 line change) |
| Team doesn't use it | Fast iteration, gather feedback |
| LangChain dependency | Industry standard, low risk |

---

## What LangChain Gives Us Free

### Immediate Benefits
- ✅ Ollama integration (no HTTP code)
- ✅ Memory management (automatic)
- ✅ Prompt engineering (templates)
- ✅ Streaming (callbacks)
- ✅ Error handling (retries)
- ✅ Testing utilities (fake LLMs)

### Future Benefits
- 🚀 RAG (vector stores ready)
- 🚀 Tools & agents (pre-built)
- 🚀 Multi-agent (LangGraph)
- 🚀 Production API (LangServe)
- 🚀 Monitoring (LangSmith)
- 🚀 Output parsing (structured data)

---

## Code Comparison

### Custom Implementation (v1.0)
```python
# OllamaClient - 100 lines
# MemoryManager - 80 lines
# PromptEngine - 50 lines
# Agent Core - 100 lines
# Error handling - 40 lines
# Streaming - 60 lines
# Testing setup - 50 lines
# -------------------------
# Total: ~480 lines
```

### LangChain Implementation (v2.0)
```python
# Agent wrapper - 20 lines
# Custom prompts - 30 lines
# Config - 25 lines
# CLI - 40 lines
# -------------------------
# Total: ~115 lines
```

**Savings: ~365 lines (76% reduction)**

---

## Next Steps

1. ✅ Architecture redesigned with LangChain
2. → Install LangChain dependencies
3. → Create project structure
4. → Implement agent wrapper (20 lines!)
5. → Add custom prompts
6. → Build CLI
7. → Test with real queries
8. → Deploy to team
9. → Iterate based on feedback

**Estimated time to working MVP: 3-5 days**

---

## Key Takeaway

### Before (Custom):
- 500+ lines of code
- 2-3 weeks development
- High maintenance
- Reinvent the wheel

### After (LangChain):
- ~100 lines of code
- 3-5 days development
- Low maintenance
- Industry-standard patterns
- Future features free

**LangChain = Smart Choice**

---

**See ARCHITECTURE.md for detailed technical design**

**Ready to build? Let's go!**
