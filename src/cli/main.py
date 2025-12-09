"""CLI interface for AI Coding Agent."""

import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ..agent.agent import CodingAgent
from ..config import settings
from ..rag.indexer import build_index
from ..rag.crawl_tracker import get_crawl_tracker
from ..agent.tools.crawl_and_index import get_crawl_and_index_tool
import asyncio

console = Console()


def print_welcome():
    """Display welcome message."""
    welcome_text = """
# 🤖 AI Coding Agent

**Model:** qwen2.5-coder:1.5b (via Ollama + LangChain)
**Expertise:** Python, TypeScript, SQL, NoSQL, OLAP databases

**Commands:**
- Type your coding question
- `tools` - List all available tools
- `config` - Show current configuration
- `history` - Show conversation history
- `index` - Build or update the RAG codebase index
- `index --force` - Force a full re-index of the codebase
- `crawl <url>` - Crawl and index a documentation URL
- `crawled` - Show crawled documentation history
- `clear` - Clear conversation history
- `info` - Show model information
- `help` - Show this help message
- `exit` or `quit` - Exit the application

**Tip:** Real-time tool execution feedback now enabled!
    """
    console.print(Panel(Markdown(welcome_text), border_style="blue"))


def print_model_info(agent: CodingAgent):
    """Display model configuration information."""
    info = agent.get_model_info()
    provider = info.get("provider", "unknown")

    table = Table(title="Model Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Common settings
    table.add_row("Provider Mode", provider.upper())
    table.add_row("Agent Mode", info.get("agent_mode", "N/A"))
    table.add_row("Temperature", str(info.get("temperature", "N/A")))

    # Ollama-specific settings
    if provider in ["ollama", "hybrid"]:
        table.add_row("─" * 20, "─" * 30)  # Separator
        table.add_row("[bold]Ollama Settings[/bold]", "")
        table.add_row("Model", info.get("ollama_model", "N/A"))
        table.add_row("Base URL", info.get("ollama_base_url", "N/A"))
        table.add_row("Deployment", info.get("ollama_deployment", "N/A").upper())

    # Claude-specific settings
    if provider in ["claude", "hybrid"]:
        table.add_row("─" * 20, "─" * 30)  # Separator
        table.add_row("[bold]Claude Settings[/bold]", "")
        table.add_row("Model", info.get("claude_model", "N/A"))
        api_status = "✓ Configured" if info.get("claude_api_configured") else "✗ Not Configured"
        table.add_row("API Key", api_status)

    # Gemini-specific settings
    if provider in ["gemini", "hybrid"]:
        table.add_row("─" * 20, "─" * 30)
        table.add_row("[bold]Gemini Settings[/bold]", "")
        table.add_row("Model", info.get("gemini_model", "N/A"))
        api_status = "✓ Configured" if info.get("gemini_api_configured") else "✗ Not Configured"
        table.add_row("API Key", api_status)

    # Hybrid-specific settings
    if provider == "hybrid":
        keywords = info.get("routing_keywords", [])
        keywords_str = ", ".join(keywords[:3]) + "..." if len(keywords) > 3 else ", ".join(keywords)
        table.add_row("─" * 20, "─" * 30)
        table.add_row("[bold]Hybrid Routing[/bold]", "")
        table.add_row("Claude Keywords", keywords_str)
        
    # Tool settings
    if info.get("agent_mode", "N/A") == "ReAct (Tools enabled)":
        table.add_row("─" * 20, "─" * 30)
        table.add_row("[bold]Enabled Tools[/bold]", ", ".join(info.get("tools", [])))

    console.print(table)


def print_help():
    """Display help message."""
    help_text = """
## Available Commands

- **index** - Build or update the RAG codebase index. This allows the agent to
  perform semantic searches over your code. Run this command whenever your
  code changes.
- **index --force** - Force a full re-index, ignoring any cached files.
- **crawl \<url\>** - Crawl and index a documentation URL. The agent can then
  search this documentation using RAG. Example: `crawl https://docs.python.org/3/library/json.html`
- **crawled** - Show history of all crawled documentation URLs with statistics.
- **clear** - Clear conversation history and start fresh.
- **info** - Show current model configuration.
- **help** - Show this help message.
- **exit** or **quit** - Exit the application.

## Example Questions

- "Write a PostgreSQL query to find duplicate emails"
- "Create a Python FastAPI endpoint with error handling"
- "Show me a MongoDB aggregation pipeline for sales data"
- "What's the best way to index a large table in MySQL?"
- "Write a TypeScript function to validate user input"

## Tips

- The agent remembers your conversation context.
- Use `clear` to start a new topic.
- Code examples are automatically syntax-highlighted.
    """
    console.print(Markdown(help_text))


def run_indexing(force: bool):
    """Run the RAG indexing process."""
    console.print("\n[yellow]Starting codebase indexing...[/yellow]")
    console.print(f"[dim]Force re-index: {force}[/dim]")
    try:
        stats = build_index(force_reindex=force)
        console.print("\n[green]✓ Indexing complete![/green]")

        table = Table(title="Indexing Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Files Indexed", str(stats.get("files_indexed", 0)))
        table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))
        table.add_row("Time Taken", f"{stats.get('time_taken', 0):.2f}s")

        console.print(table)

    except Exception as e:
        console.print(f"\n[red]✗ Indexing failed: {e}[/red]")


def run_crawl(url: str):
    """Crawl and index a documentation URL."""
    console.print(f"\n[yellow]Crawling {url}...[/yellow]")

    try:
        tool = get_crawl_and_index_tool(settings)
        result = tool._run(url)

        if "Error" in result or "Failed" in result:
            console.print(f"\n[red]{result}[/red]")
        elif "already indexed with identical content" in result:
            console.print(f"\n[blue]{result}[/blue]")
        else:
            console.print(f"\n[green]{result}[/green]")

    except Exception as e:
        console.print(f"\n[red]✗ Crawling failed: {e}[/red]")


def show_crawled_history():
    """Display crawled documentation history."""
    tracker = get_crawl_tracker()
    records = tracker.get_all_records()

    if not records:
        console.print("\n[dim]No documentation has been crawled yet[/dim]")
        console.print("[dim]Use 'crawl <url>' to index documentation[/dim]")
        return

    table = Table(title=f"Crawled Documentation ({len(records)} URLs)", show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("URL", style="cyan", width=50)
    table.add_column("Title", style="white", width=30)
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Crawled", style="yellow", width=20)

    for idx, record in enumerate(records, 1):
        # Truncate URL if too long
        url_display = record.url if len(record.url) <= 50 else record.url[:47] + "..."
        title_display = record.title if len(record.title) <= 30 else record.title[:27] + "..."

        # Format date nicely
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(record.crawl_date)
            date_display = dt.strftime("%Y-%m-%d %H:%M")
        except:
            date_display = record.crawl_date[:16]

        table.add_row(
            str(idx),
            url_display,
            title_display,
            str(record.chunk_count),
            date_display
        )

    console.print(table)

    # Show statistics
    stats = tracker.get_stats()
    console.print(f"\n[dim]Total chunks indexed: {stats['total_chunks']:,}[/dim]")
    console.print(f"[dim]Total content: {stats['total_content_bytes'] / 1024:.1f} KB[/dim]")


def main():
    """Main CLI application."""
    print_welcome()

    # Initialize the coding agent
    try:
        console.print("\n[yellow]Initializing agent...[/yellow]")
        agent = CodingAgent()
        console.print("[green]✓ Agent ready![/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize agent: {e}[/red]")
        if "ollama" in str(e).lower():
            console.print("\n[yellow]Make sure Ollama is running: `ollama serve`[/yellow]")
            console.print("[yellow]And the model is pulled, e.g., `ollama pull qwen2.5-coder:1.5b`[/yellow]")
        elif "api" in str(e).lower():
            console.print(f"\n[yellow]Check your API key in the .env file.[/yellow]")
        sys.exit(1)

    # Main conversation loop
    while True:
        try:
            # Get user input
            query = Prompt.ask("\n[bold green]You[/bold green]")

            # Handle empty input
            if not query.strip():
                continue

            # Handle commands
            if query.lower() in ['exit', 'quit']:
                console.print("\n[blue]Goodbye! 👋[/blue]")
                break
            
            elif query.lower().startswith("index"):
                force = "--force" in query.lower()
                run_indexing(force=force)
                continue

            elif query.lower().startswith("crawl "):
                url = query[6:].strip()  # Extract URL after "crawl "
                if url:
                    run_crawl(url)
                else:
                    console.print("[red]Usage: crawl <url>[/red]")
                continue

            elif query.lower() == "crawled":
                show_crawled_history()
                continue

            elif query.lower() == 'clear':
                agent.clear_history()
                console.print("[yellow]✓ Conversation history cleared[/yellow]")
                continue

            elif query.lower() == 'info':
                print_model_info(agent)
                continue

            elif query.lower() == 'help':
                print_help()
                continue

            elif query.lower() == 'tools':
                print_available_tools(agent)
                continue

            elif query.lower() == 'config':
                print_configuration()
                continue

            elif query.lower().startswith('history'):
                if 'clear' in query.lower():
                    agent.clear_history()
                    console.print("[yellow]✓ Conversation history cleared[/yellow]")
                else:
                    history = agent.get_conversation_history()
                    if history:
                        console.print("\n[bold cyan]Conversation History:[/bold cyan]")
                        console.print(history)
                    else:
                        console.print("[dim]No conversation history yet[/dim]")
                continue

            # Get response from agent with streaming
            console.print("\n[bold blue]Agent:[/bold blue]")

            response_content = ""
            for update in agent.ask_stream(query):
                update_type = update.get('type')
                content = update.get('content', '')

                if update_type == 'tool_start':
                    # Show tool execution
                    console.print(f"  [dim]{content}[/dim]", end=" ")
                elif update_type == 'tool_end':
                    # Tool completed
                    console.print(f"[green]{content}[/green]")
                elif update_type == 'response':
                    # Accumulate response for final display
                    response_content += content
                elif update_type == 'error':
                    # Show error
                    console.print(f"[red]{content}[/red]")
                    break

            # Display final response with markdown formatting
            if response_content:
                console.print(Markdown(response_content))

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            continue


def print_available_tools(agent: CodingAgent):
    """Display all available tools with descriptions."""
    table = Table(title="Available Tools", show_header=True)
    table.add_column("Tool", style="cyan", width=25)
    table.add_column("Description", style="white")

    for tool in agent.tools:
        # Truncate long descriptions
        desc = tool.description.split('\n')[0][:80]
        table.add_row(tool.name, desc)

    console.print(table)
    console.print(f"\n[dim]Total: {len(agent.tools)} tools[/dim]")


def print_configuration():
    """Display current agent configuration."""
    table = Table(title="Agent Configuration", show_header=True)
    table.add_column("Setting", style="cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("Provider", settings.LLM_PROVIDER.upper())

    if settings.LLM_PROVIDER == "gemini":
        table.add_row("Model", settings.GEMINI_MODEL)
    elif settings.LLM_PROVIDER == "claude":
        table.add_row("Model", settings.CLAUDE_MODEL)
    else:
        table.add_row("Model", settings.MODEL_NAME)

    table.add_row("Temperature", str(settings.TEMPERATURE))
    table.add_row("Max Tokens", str(settings.MAX_TOKENS))
    table.add_row("Tools Enabled", "✓" if settings.ENABLE_TOOLS else "✗")
    table.add_row("File Operations", "✓" if settings.ENABLE_FILE_OPS else "✗")
    table.add_row("RAG Search", "✓" if settings.ENABLE_RAG else "✗")

    console.print(table)


if __name__ == "__main__":
    main()
