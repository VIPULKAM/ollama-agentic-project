"""AI Coding Agent - LangChain-based implementation with multi-provider support."""

from langchain_ollama.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from typing import Optional, List
from langchain_core.tools import BaseTool

# LangGraph imports
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


from ..config.settings import settings
from .prompts import get_system_prompt, get_react_system_prompt

# Import all tool factory functions
from .tools.file_ops import (
    get_read_file_tool,
    get_write_file_tool,
    get_list_directory_tool,
    get_search_code_tool,
)
from .tools.rag_search import get_rag_search_tool


class CodingAgent:
    """AI Coding Assistant with multi-provider support (Ollama, Claude, Gemini).

    This agent supports:
    - A simple conversational chain (default)
    - A tool-using LangGraph agent (if ENABLE_TOOLS=True in settings)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the coding agent."""
        self.provider = provider or settings.LLM_PROVIDER
        self.temperature = temperature or settings.TEMPERATURE
        self.use_tools = settings.ENABLE_TOOLS

        # Initialize chat history store (used by both modes)
        self.store = {}
        self.session_id = "default"

        # Initialize providers
        self._initialize_providers(model_name, base_url, api_key)

        # Set primary LLM based on provider
        if self.provider == "claude":
            self.llm = self.claude_llm
        elif self.provider == "gemini":
            self.llm = self.gemini_llm
        else:
            # Default to Ollama for "ollama" and "hybrid" modes
            self.llm = self.ollama_llm

        if self.use_tools:
            # Setup for LangGraph Agent
            self._setup_langgraph_agent()
        else:
            # Setup for standard conversational chain
            self._setup_conversation_chain()

    def _initialize_providers(self, model_name, base_url, api_key):
        """Initializes the Ollama, Claude, and/or Gemini LLM providers."""
        self.ollama_llm: Optional[BaseLanguageModel] = None
        self.claude_llm: Optional[BaseLanguageModel] = None
        self.gemini_llm: Optional[BaseLanguageModel] = None

        if self.provider in ["ollama", "hybrid"]:
            self.model_name = model_name or settings.MODEL_NAME
            self.base_url = base_url or settings.OLLAMA_BASE_URL
            self.ollama_llm = ChatOllama(
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
            )

        if self.provider in ["claude", "hybrid"]:
            claude_api_key = api_key if self.provider == "claude" else None
            claude_api_key = claude_api_key or settings.ANTHROPIC_API_KEY
            if not claude_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY must be set in .env for Claude provider"
                )
            self.claude_model = settings.CLAUDE_MODEL
            self.claude_llm = ChatAnthropic(
                api_key=claude_api_key,
                model=self.claude_model,
                temperature=self.temperature,
                max_tokens=settings.MAX_TOKENS,
            )

        if self.provider in ["gemini", "hybrid"]:
            gemini_api_key = api_key if self.provider == "gemini" else None
            gemini_api_key = gemini_api_key or settings.GOOGLE_API_KEY
            if not gemini_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY must be set in .env for Gemini provider"
                )
            self.gemini_model = settings.GEMINI_MODEL
            self.gemini_llm = ChatGoogleGenerativeAI(
                google_api_key=gemini_api_key,
                model=self.gemini_model,
                temperature=self.temperature,
                max_tokens=settings.MAX_TOKENS,
            )

    def _setup_conversation_chain(self):
        """Sets up the basic conversational chain without tools."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", get_system_prompt()),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _setup_langgraph_agent(self):
        """Sets up the LangGraph agent with tools using prebuilt create_react_agent."""
        # 1. Gather tools
        self.tools: List[BaseTool] = []
        if settings.ENABLE_FILE_OPS:
            self.tools.extend([
                get_read_file_tool(),
                get_write_file_tool(),
                get_list_directory_tool(),
                get_search_code_tool(),
            ])
        if settings.ENABLE_RAG:
            self.tools.append(get_rag_search_tool(settings))

        # 2. Create the ReAct agent using LangGraph's prebuilt function
        # This handles all the complexity of tool calling, message routing, etc.
        self.checkpointer = MemorySaver()
        self.langgraph_app = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=self.checkpointer
        )

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Get or create session history."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def _should_use_claude(self, query: str) -> bool:
        """Determine if query should use Claude in hybrid mode."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in settings.CLAUDE_KEYWORDS)

    def ask(self, query: str, force_provider: Optional[str] = None) -> BaseMessage:
        """Ask the agent a question. Dispatches to the appropriate chain/agent."""
        try:
            selected_llm = self._select_llm(query, force_provider)

            if self.use_tools:
                # Rebuild agent if LLM changed (for hybrid mode)
                if self.llm != selected_llm:
                    self.llm = selected_llm
                    self._setup_langgraph_agent()  # Rebuild with new LLM

                # Get conversation history
                history = self._get_session_history(self.session_id).messages

                # Create input with history + new query
                from langchain_core.messages import HumanMessage
                messages = history + [HumanMessage(content=query)]

                # Invoke LangGraph with checkpointing for conversation memory
                config = {"configurable": {"thread_id": self.session_id}}
                result = self.langgraph_app.invoke(
                    {"messages": messages},
                    config=config
                )

                # Extract the final AI response
                final_message = result["messages"][-1]

                # Update conversation history
                # Add the user query
                self._get_session_history(self.session_id).add_user_message(query)
                # Add the AI response
                if isinstance(final_message, AIMessage):
                    self._get_session_history(self.session_id).add_ai_message(final_message.content)

                return final_message
            else:
                return self._ask_conversation_chain(query, selected_llm)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\n"
            if "ollama" in str(e).lower():
                error_msg += "Make sure Ollama is running: `ollama serve`"
            elif "anthropic" in str(e).lower():
                error_msg += "Check your ANTHROPIC_API_KEY in .env file"
            elif "gemini" in str(e).lower() or "google" in str(e).lower():
                error_msg += "Check your GOOGLE_API_KEY in .env file"
            return AIMessage(content=error_msg)

    def _select_llm(self, query: str, force_provider: Optional[str]) -> BaseLanguageModel:
        """Selects the appropriate LLM based on provider settings and query."""
        selected_llm = self.llm

        if force_provider:
            if force_provider == "claude" and self.claude_llm:
                selected_llm = self.claude_llm
            elif force_provider == "gemini" and self.gemini_llm:
                selected_llm = self.gemini_llm
            elif force_provider == "ollama" and self.ollama_llm:
                selected_llm = self.ollama_llm
        elif self.provider == "hybrid":
            if self._should_use_claude(query) and self.claude_llm:
                selected_llm = self.claude_llm
                print(f"\n[Hybrid Mode: Using Claude for '{query[:30]}...']")
            else:
                selected_llm = self.ollama_llm
                print(f"\n[Hybrid Mode: Using Ollama for '{query[:30]}...']")

        return selected_llm

    def _rebuild_chain_with_llm(self, llm: BaseLanguageModel) -> RunnableWithMessageHistory:
        """Rebuild the conversational chain with a different LLM (used in hybrid mode)."""
        chain = self.prompt | llm
        return RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )



    def _ask_conversation_chain(self, query: str, llm: BaseLanguageModel) -> BaseMessage:
        """Invokes the basic conversational chain."""
        chain_with_history = self._rebuild_chain_with_llm(llm)
        response = chain_with_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": self.session_id}}
        )
        # Ensure the output is always a BaseMessage
        return response if isinstance(response, BaseMessage) else AIMessage(content=str(response))




    def clear_history(self) -> None:
        """Clear the conversation history."""
        if self.session_id in self.store:
            self.store[self.session_id].clear()

    def get_conversation_history(self) -> str:
        """Get the conversation history as a string."""
        if self.session_id in self.store:
            history = self.store[self.session_id]
            messages = []
            for msg in history.messages:
                # Handle both string and BaseMessage content
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if msg.type == "human":
                    messages.append(f"User: {content}")
                elif msg.type == "ai":
                    messages.append(f"Agent: {content}")
                else: # Fallback for other message types
                    messages.append(f"{msg.type.capitalize()}: {content}")

            return "\n".join(messages)
        return ""

    def get_model_info(self) -> dict:
        """Get information about the current model configuration."""
        info = {
            "provider": self.provider,
            "temperature": self.temperature,
            "agent_mode": "LangGraph (Tools enabled)" if self.use_tools else "Conversational",
        }

        if self.provider in ["ollama", "hybrid"]:
            info.update({
                "ollama_model": getattr(self, 'model_name', None),
                "ollama_base_url": getattr(self, 'base_url', None),
                "ollama_deployment": "local" if "localhost" in getattr(self, 'base_url', '') else "cloud",
            })

        if self.provider in ["claude", "hybrid"]:
            info.update({
                "claude_model": getattr(self, 'claude_model', None),
                "claude_api_configured": bool(getattr(self, 'api_key', None)),
            })
        
        if self.provider in ["gemini", "hybrid"]:
            info.update({
                "gemini_model": getattr(self, 'gemini_model', None),
                "gemini_api_configured": bool(getattr(self, 'gemini_llm', None)),
            })

        if self.provider == "hybrid":
            info["routing_keywords"] = settings.CLAUDE_KEYWORDS
        
        if self.use_tools:
            info["tools"] = [tool.name for tool in self.tools]

        return info
