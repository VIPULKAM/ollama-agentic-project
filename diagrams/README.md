# Architecture Diagrams

This folder contains visual representations of the AI Coding Agent architecture.

## 📊 Available Diagrams

### ✅ Complete (PNG + SVG)
1. **Overall System Architecture** - High-level system overview
2. **LangGraph Agent Execution Flow** - Step-by-step agent processing
3. **Multi-Provider System** - Ollama/Claude/Gemini routing logic
4. **RAG System Architecture** - Indexing and retrieval pipeline
6. **Conversation Memory Flow** - History and state management
10. **Complete Request Lifecycle** - End-to-end sequence diagram

### ⚠️ SVG Only (PNG generation failed)
5. **Tool Execution Flow** - State machine for tool calling
7. **Configuration System** - Settings and environment flow
8. **Error Handling & Fallback** - Error recovery mechanisms
9. **File Operation Security** - Security validation flow

---

## 🎨 File Formats

### PNG Files (~9KB each)
- **Use for**: Documentation, presentations, quick viewing
- **Pros**: Universal compatibility, easy to embed
- **Cons**: Fixed resolution, larger file size

### SVG Files (~3.4KB each)
- **Use for**: Web, scalable graphics, high-quality prints
- **Pros**: Infinitely scalable, small file size, editable
- **Cons**: May not display in all viewers

---

## 📖 How to View

### PNG Files
```bash
# Linux/Mac
open 01_Overall_System_Architecture.png

# Or use any image viewer
eog *.png
```

### SVG Files
```bash
# Open in browser
firefox 01_Overall_System_Architecture.svg

# Or use Inkscape (vector graphics editor)
inkscape 01_Overall_System_Architecture.svg
```

---

## 🔄 Regenerating Diagrams

If you need to regenerate or update diagrams:

```bash
# From project root
python generate_diagrams.py
```

This will:
1. Extract diagrams from `ARCHITECTURE_DIAGRAMS.md`
2. Use mermaid.ink API to generate images
3. Save PNG and SVG files in `diagrams/` folder

---

## 📝 Diagram Descriptions

### 1. Overall System Architecture
Shows the complete system with all layers:
- User Interface (CLI, Future Web UI)
- Application Layer (Agent, Settings)
- LLM Providers (Ollama, Claude, Gemini)
- Agent Modes (Simple, LangGraph)
- Tools System (5 tools)
- RAG System (Embeddings, FAISS, Retriever)
- Storage (File System, Vector DB, Memory)

### 2. LangGraph Agent Execution Flow
Sequence diagram showing:
- User query processing
- Memory loading and saving
- ReAct loop with tool calls
- LLM decision-making
- Tool execution and results

### 3. Multi-Provider System
Decision flow for provider selection:
- Ollama (local, free)
- Claude (API, paid)
- Gemini (API, free tier)
- Hybrid mode with keyword routing

### 4. RAG System Architecture
Two-phase process:
- **Indexing Phase**: File discovery → Chunking → Embeddings → FAISS index
- **Retrieval Phase**: Query → Embedding → Search → Ranked results → LLM

### 5. Tool Execution Flow
State machine showing:
- Agent start
- LLM decision (tool or answer)
- Tool execution (read, write, list, search, RAG)
- Result processing
- Loop or final answer

### 6. Conversation Memory Flow
Shows how conversation context is maintained:
- Session management with IDs
- InMemory store for history
- LangGraph checkpointing with threads
- Message loading and saving

### 7. Configuration System
Configuration flow from sources to components:
- .env file, environment variables, defaults
- Pydantic Settings class
- Provider, tool, RAG, file settings
- Used by Agent, Tools, RAG system

### 8. Error Handling & Fallback
Error handling decision tree:
- Provider availability checks
- API key validation
- Credit/quota verification
- User-friendly error messages

### 9. File Operation Security
Security validation flow:
- Path traversal detection
- File size limits (10MB)
- Backup creation for writes
- Logging and error handling

### 10. Complete Request Lifecycle
Full sequence diagram with all interactions:
- User → CLI → Agent
- Configuration loading
- LLM selection (hybrid mode)
- Memory management
- LangGraph/Simple mode branching
- Tool execution loop
- Response formatting

---

## 🎨 Color Coding

Diagrams use consistent color coding:

| Color | Component Type |
|-------|----------------|
| 🟢 Green | Success states, Ollama provider |
| 🔴 Red | Error states, Claude provider |
| 🔵 Blue | LangGraph, processing, Gemini provider |
| 🟣 Purple | RAG system, FAISS operations |
| 🟠 Orange | Tools, embeddings |
| 🟡 Yellow | Decision points, routing |

---

## 💡 Tips

### For Presentations
- Use PNG files for slides (PowerPoint, Google Slides)
- Scale to fit without losing quality

### For Documentation
- Use SVG files in markdown/HTML
- Better quality at any zoom level

### For Editing
- SVG files can be edited in:
  - Inkscape (free, open-source)
  - Adobe Illustrator
  - Figma (import SVG)

### For Printing
- Use SVG files for high-quality prints
- PNG files work but may pixelate when enlarged

---

## 🔗 Related Files

- **ARCHITECTURE_DIAGRAMS.md** - Source markdown with Mermaid code
- **generate_diagrams.py** - Script to generate these images
- **CLAUDE.md** - Detailed technical documentation

---

## ⚡ Quick Commands

```bash
# View all PNGs in system viewer
xdg-open *.png

# Open all SVGs in browser
firefox *.svg

# Create a collage (requires ImageMagick)
montage *.png -geometry 400x400+10+10 -tile 2x5 collage.png

# Convert SVG to high-res PNG (requires Inkscape)
for f in *.svg; do inkscape "$f" --export-png="${f%.svg}_hires.png" --export-dpi=300; done
```

---

Generated on: 2025-12-04
Total files: 16 (7 PNG, 9 SVG)
Total size: ~112KB
