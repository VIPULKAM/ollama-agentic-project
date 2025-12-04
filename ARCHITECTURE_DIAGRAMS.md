# Architecture Diagrams - AI Coding Agent

This document contains Mermaid diagrams visualizing the project architecture and data flows.

---

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[CLI Interface<br/>Rich Terminal UI]
        Future[Future: Web UI]
    end

    subgraph "Application Layer"
        Agent[CodingAgent<br/>Main Orchestrator]
        Settings[Settings<br/>Pydantic Config]
    end

    subgraph "LLM Provider Layer"
        Ollama[Ollama LLM<br/>Self-hosted]
        Claude[Claude API<br/>Anthropic]
        Gemini[Gemini API<br/>Google]
    end

    subgraph "Agent Modes"
        Simple[Simple Chain<br/>Conversational]
        LangGraph[LangGraph Agent<br/>Tool-based ReAct]
    end

    subgraph "Tools System"
        ReadFile[Read File Tool]
        WriteFile[Write File Tool]
        ListDir[List Directory Tool]
        SearchCode[Search Code Tool]
        RAGSearch[RAG Search Tool]
    end

    subgraph "RAG System"
        Embeddings[Embeddings<br/>sentence-transformers]
        Indexer[FAISS Indexer<br/>Vector Store]
        Retriever[Semantic Retriever]
    end

    subgraph "Storage"
        FileSystem[(File System)]
        VectorDB[(FAISS Index<br/>~/.ai-agent/)]
        Memory[(In-Memory History)]
    end

    CLI --> Agent
    Future -.-> Agent
    Agent --> Settings
    Settings -.-> Agent

    Agent --> Simple
    Agent --> LangGraph

    Simple --> Ollama
    Simple --> Claude
    Simple --> Gemini

    LangGraph --> Ollama
    LangGraph --> Claude
    LangGraph --> Gemini

    LangGraph --> ReadFile
    LangGraph --> WriteFile
    LangGraph --> ListDir
    LangGraph --> SearchCode
    LangGraph --> RAGSearch

    ReadFile --> FileSystem
    WriteFile --> FileSystem
    ListDir --> FileSystem
    SearchCode --> FileSystem
    RAGSearch --> Retriever

    Retriever --> Indexer
    Indexer --> VectorDB
    Embeddings --> Indexer

    Agent --> Memory

    style Agent fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style LangGraph fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style RAGSearch fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff
    style Indexer fill:#9C27B0,stroke:#6A1B9A,stroke-width:2px,color:#fff
```

---

## 2. LangGraph Agent Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LangGraph
    participant LLM
    participant Tools
    participant Memory

    User->>Agent: ask("Read file X")
    Agent->>Memory: Load conversation history
    Memory-->>Agent: Previous messages

    Agent->>LangGraph: invoke({messages: [...history, query]})

    loop ReAct Loop
        LangGraph->>LLM: Process messages + tool definitions
        LLM-->>LangGraph: Decision (answer or tool call)

        alt Tool Call Needed
            LangGraph->>Tools: Execute tool(args)
            Tools-->>LangGraph: Tool result
            LangGraph->>LLM: Continue with tool result
        else Final Answer
            LangGraph-->>Agent: Final AI message
        end
    end

    Agent->>Memory: Save query + response
    Agent-->>User: Response content
```

---

## 3. Multi-Provider System

```mermaid
graph LR
    subgraph "Provider Selection"
        Query[User Query]
        Router{Provider Mode?}
    end

    subgraph "Hybrid Mode Logic"
        Keywords{Contains<br/>Claude Keywords?}
    end

    subgraph "LLM Providers"
        OllamaLLM[Ollama<br/>qwen2.5-coder:1.5b<br/>Local/Free]
        ClaudeLLM[Claude<br/>claude-3-haiku<br/>API/$$$]
        GeminiLLM[Gemini<br/>gemini-1.5-flash<br/>API/Free Tier]
    end

    subgraph "Response"
        Answer[AI Response]
    end

    Query --> Router

    Router -->|ollama| OllamaLLM
    Router -->|claude| ClaudeLLM
    Router -->|gemini| GeminiLLM
    Router -->|hybrid| Keywords

    Keywords -->|Yes:<br/>architecture,<br/>design,<br/>refactor| ClaudeLLM
    Keywords -->|No| OllamaLLM

    OllamaLLM --> Answer
    ClaudeLLM --> Answer
    GeminiLLM --> Answer

    style ClaudeLLM fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px
    style GeminiLLM fill:#4DABF7,stroke:#1864AB,stroke-width:2px
    style OllamaLLM fill:#51CF66,stroke:#2B8A3E,stroke-width:2px
    style Keywords fill:#FFD93D,stroke:#F59F00,stroke-width:2px
```

---

## 4. RAG System Architecture

```mermaid
graph TB
    subgraph "Indexing Phase (Offline)"
        Codebase[Codebase<br/>Python, JS, TS, etc.]
        Discovery[File Discovery<br/>+ .gitignore filtering]
        Chunker[Code Chunker<br/>AST-based]
        TextChunker[Text Chunker<br/>Sliding window]
        EmbedGen[Embedding Generator<br/>all-MiniLM-L6-v2]
        FAISSIndex[(FAISS Index<br/>IndexFlatL2)]

        Codebase --> Discovery
        Discovery --> Chunker
        Discovery --> TextChunker

        Chunker -->|Functions/Classes<br/>with context| EmbedGen
        TextChunker -->|500 char chunks<br/>50 char overlap| EmbedGen

        EmbedGen --> FAISSIndex
    end

    subgraph "Retrieval Phase (Runtime)"
        Query[User Query]
        QueryEmbed[Query Embedding]
        Search[Semantic Search<br/>top_k=5, threshold=0.5]
        Results[Ranked Results<br/>with scores]
        LLM[LLM Context]

        Query --> QueryEmbed
        QueryEmbed --> Search
        FAISSIndex --> Search
        Search --> Results
        Results --> LLM
    end

    style FAISSIndex fill:#9C27B0,stroke:#6A1B9A,stroke-width:3px,color:#fff
    style EmbedGen fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff
    style Search fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
```

---

## 5. Tool Execution Flow

```mermaid
stateDiagram-v2
    [*] --> AgentStart: User Query

    AgentStart --> LLMDecision: Process with tools available

    LLMDecision --> UseTool: Tool call needed
    LLMDecision --> FinalAnswer: Direct answer

    UseTool --> ReadFile: read_file
    UseTool --> WriteFile: write_file
    UseTool --> ListDir: list_directory
    UseTool --> SearchCode: search_code
    UseTool --> RAGSearch: rag_search

    ReadFile --> ToolResult: File content
    WriteFile --> ToolResult: Success/Backup created
    ListDir --> ToolResult: Directory listing
    SearchCode --> ToolResult: Search matches
    RAGSearch --> ToolResult: Semantic matches

    ToolResult --> LLMDecision: Continue with result

    FinalAnswer --> SaveHistory: Update memory
    SaveHistory --> [*]: Return to user

    note right of UseTool
        All tools have:
        - Path validation
        - Security checks
        - Logging
        - Error handling
    end note

    note right of RAGSearch
        Requires FAISS index
        Uses embeddings
        Semantic similarity
    end note
```

---

## 6. Conversation Memory Flow

```mermaid
graph LR
    subgraph "Session Management"
        SessionID[Session ID<br/>default]
        Store[(InMemory Store<br/>Dict[str, History])]
    end

    subgraph "Message Flow"
        UserMsg[User Message]
        History[Load History]
        Processing[LLM Processing]
        AIMsg[AI Response]
        Save[Save to History]
    end

    subgraph "LangGraph Memory"
        Checkpointer[MemorySaver<br/>Checkpointer]
        ThreadID[Thread ID]
    end

    SessionID --> Store
    UserMsg --> History
    Store --> History
    History --> Processing
    Processing --> AIMsg
    AIMsg --> Save
    Save --> Store

    ThreadID --> Checkpointer
    Processing --> Checkpointer
    Checkpointer --> Processing

    style Store fill:#4CAF50,stroke:#2E7D32,stroke-width:2px,color:#fff
    style Checkpointer fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
```

---

## 7. Configuration System

```mermaid
graph TB
    subgraph "Configuration Sources"
        EnvFile[.env File]
        EnvVars[Environment Variables]
        Defaults[Default Values]
    end

    subgraph "Settings Class (Pydantic)"
        CoreSettings[Core Settings<br/>Provider, Model, Temp]
        ProviderSettings[Provider Settings<br/>Ollama, Claude, Gemini]
        ToolSettings[Tool Settings<br/>Enable flags]
        RAGSettings[RAG Settings<br/>FAISS, Embeddings]
        FileSettings[File Settings<br/>Size limits, backups]
    end

    subgraph "Components"
        Agent[CodingAgent]
        Tools[Tool Instances]
        RAGSystem[RAG System]
    end

    EnvFile --> CoreSettings
    EnvVars --> CoreSettings
    Defaults --> CoreSettings

    CoreSettings --> ProviderSettings
    CoreSettings --> ToolSettings
    CoreSettings --> RAGSettings
    CoreSettings --> FileSettings

    ProviderSettings --> Agent
    ToolSettings --> Tools
    RAGSettings --> RAGSystem
    FileSettings --> Tools

    style CoreSettings fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    style ToolSettings fill:#4DABF7,stroke:#1864AB,stroke-width:2px,color:#fff
    style RAGSettings fill:#9C27B0,stroke:#6A1B9A,stroke-width:2px,color:#fff
```

---

## 8. Error Handling & Fallback

```mermaid
graph TD
    Request[User Request]

    Request --> CheckProvider{Provider<br/>Available?}

    CheckProvider -->|Ollama| CheckOllama{Ollama<br/>Running?}
    CheckProvider -->|Claude| CheckClaude{API Key<br/>Valid?}
    CheckProvider -->|Gemini| CheckGemini{API Key<br/>Valid?}

    CheckOllama -->|Yes| OllamaOK[Execute with Ollama]
    CheckOllama -->|No| OllamaError[Error: Start ollama serve]

    CheckClaude -->|Yes| CheckCredits{Has<br/>Credits?}
    CheckClaude -->|No| ClaudeError[Error: Add API key]

    CheckCredits -->|Yes| ClaudeOK[Execute with Claude]
    CheckCredits -->|No| CreditError[Error: Add credits]

    CheckGemini -->|Yes| GeminiOK[Execute with Gemini]
    CheckGemini -->|No| GeminiError[Error: Invalid/Expired key]

    OllamaOK --> Success[Return Response]
    ClaudeOK --> Success
    GeminiOK --> Success

    OllamaError --> ErrorMsg[User-Friendly Error]
    ClaudeError --> ErrorMsg
    CreditError --> ErrorMsg
    GeminiError --> ErrorMsg

    ErrorMsg --> User[Display to User]

    style Success fill:#51CF66,stroke:#2B8A3E,stroke-width:2px,color:#fff
    style ErrorMsg fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
```

---

## 9. File Operation Security

```mermaid
graph TB
    ToolCall[File Operation Request]

    ToolCall --> ValidatePath{Path<br/>Validation}

    ValidatePath -->|Invalid| RejectPath[Reject: Path traversal detected]
    ValidatePath -->|Valid| CheckSize{Check<br/>File Size}

    CheckSize -->|Too Large| RejectSize[Reject: Exceeds 10MB limit]
    CheckSize -->|OK| CheckBackup{Backup<br/>Needed?}

    CheckBackup -->|Write Op| CreateBackup[Create .bak file]
    CheckBackup -->|Read Op| Proceed[Proceed]

    CreateBackup --> Proceed

    Proceed --> Execute[Execute Operation]
    Execute --> Log[Log Operation]
    Log --> Success[Return Result]

    RejectPath --> Error[Return Error]
    RejectSize --> Error

    style Success fill:#51CF66,stroke:#2B8A3E,stroke-width:2px,color:#fff
    style Error fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    style CreateBackup fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#000
```

---

## 10. Complete Request Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant CLI as CLI Interface
    participant A as CodingAgent
    participant S as Settings
    participant LG as LangGraph
    participant LLM as LLM Provider
    participant T as Tools
    participant M as Memory
    participant FS as File System

    U->>CLI: Type query + Enter
    CLI->>A: ask(query)
    A->>S: Load configuration
    S-->>A: Provider, settings

    A->>A: Select LLM based on provider/mode
    A->>M: Get conversation history
    M-->>A: Previous messages

    alt Tools Enabled
        A->>LG: invoke(messages, config)

        loop ReAct Loop
            LG->>LLM: Messages + tool definitions
            LLM-->>LG: Response (tool call or answer)

            opt Tool Call
                LG->>T: Execute tool
                T->>FS: File operation
                FS-->>T: Result
                T-->>LG: Tool output
            end
        end

        LG-->>A: Final response
    else Simple Mode
        A->>LLM: Messages (no tools)
        LLM-->>A: Response
    end

    A->>M: Save conversation
    A->>CLI: Return response
    CLI->>U: Display formatted output
```

---

## How to View These Diagrams

### Option 1: GitHub (Recommended)
- Push this file to GitHub
- View in browser - GitHub renders Mermaid automatically

### Option 2: VS Code
- Install "Markdown Preview Mermaid Support" extension
- Open this file
- Click preview button (Ctrl+Shift+V)

### Option 3: Online Viewer
- Visit: https://mermaid.live/
- Copy-paste any diagram code
- View rendered output

### Option 4: Command Line
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate PNG from this file
mmdc -i ARCHITECTURE_DIAGRAMS.md -o architecture.png
```

---

## Diagram Key

| Color | Meaning |
|-------|---------|
| 🟢 Green | Success states, working components |
| 🔴 Red | Error states, critical components |
| 🔵 Blue | LangGraph, processing, retrieval |
| 🟣 Purple | RAG system, vector operations |
| 🟠 Orange | Tools, embeddings |
| 🟡 Yellow | Decision points, hybrid routing |

---

## Legend

- **Solid arrows** (→): Direct flow/dependency
- **Dashed arrows** (⇢): Optional/future feature
- **Thick borders**: Critical components
- **Subgraphs**: Logical grouping of components
- **Diamonds**: Decision points
- **Cylinders**: Data storage
- **Rectangles**: Processing units
