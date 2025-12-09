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
from ..rag.index_manager import (
    get_index_info,
    rebuild_index,
    clean_index,
    search_index,
    IndexNotFoundError
)
from ..rag.batch_crawler import (
    batch_crawl_from_file,
    batch_crawl_from_sitemap,
    BatchCrawlResult
)
import asyncio
from pathlib import Path

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
- **Index Management:**
  - `index` or `index --info` - Show index statistics
  - `index --rebuild` - Rebuild entire index from scratch
  - `index --clean` - Remove orphaned chunks
  - `index --search "query" [--threshold 0.3]` - Search the index directly
- **Documentation Crawling:**
  - `crawl <url>` - Crawl and index a single documentation URL
  - `crawl --batch <file> [--parallel N]` - Batch crawl URLs from file
  - `crawl --sitemap <url> [--filter pattern] [--parallel N]` - Crawl from sitemap
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

### Index Management
- **index** or **index --info** - Show comprehensive index statistics (chunks, files, size, breakdown).
- **index --rebuild** - Rebuild the entire index from scratch. Re-scans all local files.
- **index --clean** - Remove orphaned chunks from deleted files. Keeps index clean and efficient.
- **index --search "query"** - Search the index directly and see top results with scores.
  - Add `--source local` or `--source crawled` to filter by source
  - Add `--top N` to control number of results (default: 10)
  - Add `--threshold 0.0-1.0` to set minimum similarity (default: 0.5, lower = more results)
  - Example: `index --search "async function" --source local --top 5 --threshold 0.3`

### Documentation Crawling
- **crawl \<url\>** - Crawl and index a documentation URL. The agent can then
  search this documentation using RAG. Example: `crawl https://docs.python.org/3/library/json.html`
- **crawled** - Show history of all crawled documentation URLs with statistics.

### General Commands
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


def show_index_info():
    """Display comprehensive index statistics."""
    try:
        info = get_index_info()

        if not info["exists"]:
            console.print(f"\n[yellow]{info['message']}[/yellow]")
            return

        # Create main statistics table
        table = Table(title="📊 Index Statistics", show_header=True)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green")

        table.add_row("Total Chunks", f"{info['total_chunks']:,}")
        table.add_row("Total Files", f"{info['total_files']:,}")
        table.add_row("Crawled URLs", f"{info['crawled_urls']:,}")
        table.add_row("Index Size", f"{info['index_size_mb']} MB")
        table.add_row("Last Updated", info['last_updated'])
        table.add_row("Embedding Model", info['embedding_model'])

        console.print("\n", table)

        # Breakdown by source
        source_table = Table(title="Chunks by Source", show_header=True)
        source_table.add_column("Source", style="cyan")
        source_table.add_column("Chunks", style="green", justify="right")

        for source, count in info['chunks_by_source'].items():
            source_table.add_row(source.capitalize(), f"{count:,}")

        console.print("\n", source_table)

        # Breakdown by language (top 5)
        lang_table = Table(title="Top Languages", show_header=True)
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Chunks", style="green", justify="right")

        for lang, count in list(info['chunks_by_language'].items())[:5]:
            lang_table.add_row(lang, f"{count:,}")

        console.print("\n", lang_table)

    except IndexNotFoundError as e:
        console.print(f"\n[red]✗ {e}[/red]")
    except Exception as e:
        console.print(f"\n[red]✗ Error getting index info: {e}[/red]")


def run_index_rebuild():
    """Rebuild the entire index from scratch."""
    console.print("\n[yellow]⚠️  This will rebuild the entire index from scratch.[/yellow]")
    console.print("[dim]All local files will be re-indexed. Crawled docs will be preserved.[/dim]\n")

    confirm = Prompt.ask("Continue?", choices=["y", "n"], default="n")

    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    console.print("\n[yellow]Rebuilding index...[/yellow]")

    try:
        result = rebuild_index(show_progress=True)

        if result["success"]:
            console.print(f"\n[green]✓ Index rebuilt successfully![/green]")

            table = Table(title="Rebuild Statistics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Chunks Indexed", f"{result['chunks_indexed']:,}")
            table.add_row("Files Processed", f"{result['files_processed']:,}")
            table.add_row("Duration", f"{result['duration_seconds']}s")

            console.print(table)
        else:
            console.print(f"\n[red]✗ Rebuild failed[/red]")
            for error in result["errors"]:
                console.print(f"[red]  - {error}[/red]")

    except Exception as e:
        console.print(f"\n[red]✗ Rebuild failed: {e}[/red]")


def run_index_clean():
    """Clean orphaned chunks from the index."""
    console.print("\n[yellow]Cleaning orphaned chunks...[/yellow]")
    console.print("[dim]This will remove chunks from deleted files.[/dim]\n")

    try:
        result = clean_index()

        if result["chunks_removed"] == 0:
            console.print("\n[green]✓ Index is already clean![/green]")
            console.print(f"[dim]{result['chunks_remaining']:,} chunks remain in index[/dim]")
        else:
            console.print(f"\n[green]✓ Index cleaned successfully![/green]")

            table = Table(title="Cleaning Results", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Chunks Removed", f"{result['chunks_removed']:,}")
            table.add_row("Chunks Remaining", f"{result['chunks_remaining']:,}")
            table.add_row("Files Removed", str(len(result['files_removed'])))

            console.print(table)

            if result['files_removed'] and len(result['files_removed']) <= 10:
                console.print("\n[dim]Removed files:[/dim]")
                for file in result['files_removed'][:10]:
                    console.print(f"[dim]  - {file}[/dim]")

    except IndexNotFoundError as e:
        console.print(f"\n[yellow]{e}[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Cleaning failed: {e}[/red]")


def run_index_search(query: str, top_k: int = 10, source_filter: str = None, threshold: float = 0.5):
    """Search the index directly."""
    console.print(f"\n[yellow]Searching for: \"{query}\"[/yellow]")
    if source_filter:
        console.print(f"[dim]Filter: {source_filter} files only[/dim]")
    console.print(f"[dim]Threshold: {threshold} | Top: {top_k}[/dim]")

    try:
        results = search_index(query, top_k=top_k, threshold=threshold, source_filter=source_filter)

        if not results:
            console.print("\n[yellow]No results found[/yellow]")
            return

        console.print(f"\n[green]Found {len(results)} results:[/green]\n")

        for i, result in enumerate(results, 1):
            # Create result panel
            score_color = "green" if result["score"] > 0.7 else "yellow" if result["score"] > 0.5 else "red"
            title = f"#{i} - {result['file_path']} (Score: [{score_color}]{result['score']:.3f}[/{score_color}])"

            content = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]

            info = f"[dim]Language: {result.get('language', 'unknown')}"
            if result.get('start_line'):
                info += f" | Lines: {result['start_line']}-{result['end_line']}"
            info += "[/dim]"

            console.print(Panel(f"{content}\n\n{info}", title=title, border_style="blue"))

    except IndexNotFoundError as e:
        console.print(f"\n[yellow]{e}[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Search failed: {e}[/red]")


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


def run_batch_crawl(file_path: str, max_concurrent: int = 5):
    """Crawl multiple URLs from a file."""
    console.print(f"\n[yellow]Batch crawling URLs from {file_path}...[/yellow]")
    console.print(f"[dim]Max concurrent: {max_concurrent}[/dim]\n")

    try:
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]✗ File not found: {file_path}[/red]")
            return

        # Run async batch crawl
        result = asyncio.run(batch_crawl_from_file(
            path,
            max_concurrent=max_concurrent,
            show_progress=True
        ))

        # Display results
        console.print(f"\n[green]✓ Batch crawl complete![/green]")

        table = Table(title="Batch Crawl Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total URLs", str(result.total_urls))
        table.add_row("Successful", f"[green]{result.successful}[/green]")
        table.add_row("Failed", f"[red]{result.failed}[/red]" if result.failed > 0 else "0")
        table.add_row("Skipped (already crawled)", f"[blue]{result.skipped}[/blue]")
        table.add_row("Duration", f"{result.duration_seconds:.1f}s")

        console.print(table)

        # Show failed URLs if any
        if result.urls_failed:
            console.print("\n[red]Failed URLs:[/red]")
            for failed in result.urls_failed[:5]:  # Show first 5
                console.print(f"[red]  ✗ {failed['url']}: {failed['error'][:50]}...[/red]")
            if len(result.urls_failed) > 5:
                console.print(f"[dim]  ... and {len(result.urls_failed) - 5} more[/dim]")

    except Exception as e:
        console.print(f"\n[red]✗ Batch crawl failed: {e}[/red]")


def run_sitemap_crawl(sitemap_url: str, url_filter: str = None, max_concurrent: int = 5):
    """Crawl URLs from a sitemap."""
    console.print(f"\n[yellow]Crawling from sitemap: {sitemap_url}...[/yellow]")
    if url_filter:
        console.print(f"[dim]Filter: Only URLs containing '{url_filter}'[/dim]")
    console.print(f"[dim]Max concurrent: {max_concurrent}[/dim]\n")

    try:
        # Run async sitemap crawl
        result = asyncio.run(batch_crawl_from_sitemap(
            sitemap_url,
            url_filter=url_filter,
            max_concurrent=max_concurrent,
            show_progress=True
        ))

        # Display results
        console.print(f"\n[green]✓ Sitemap crawl complete![/green]")

        table = Table(title="Sitemap Crawl Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total URLs", str(result.total_urls))
        table.add_row("Successful", f"[green]{result.successful}[/green]")
        table.add_row("Failed", f"[red]{result.failed}[/red]" if result.failed > 0 else "0")
        table.add_row("Skipped (already crawled)", f"[blue]{result.skipped}[/blue]")
        table.add_row("Duration", f"{result.duration_seconds:.1f}s")

        console.print(table)

        # Show failed URLs if any
        if result.urls_failed:
            console.print("\n[red]Failed URLs:[/red]")
            for failed in result.urls_failed[:5]:
                console.print(f"[red]  ✗ {failed['url']}: {failed['error'][:50]}...[/red]")
            if len(result.urls_failed) > 5:
                console.print(f"[dim]  ... and {len(result.urls_failed) - 5} more[/dim]")

    except Exception as e:
        console.print(f"\n[red]✗ Sitemap crawl failed: {e}[/red]")


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
                # Parse index subcommands
                parts = query.lower().split()

                if len(parts) == 1 or (len(parts) == 2 and parts[1] == "--info"):
                    # index or index --info
                    show_index_info()
                elif len(parts) >= 2 and parts[1] == "--rebuild":
                    # index --rebuild
                    run_index_rebuild()
                elif len(parts) >= 2 and parts[1] == "--clean":
                    # index --clean
                    run_index_clean()
                elif len(parts) >= 2 and parts[1] == "--search":
                    # index --search "query" [--source local|crawled] [--top 10] [--threshold 0.3]
                    if len(parts) < 3:
                        console.print("[red]Usage: index --search \"query\" [--source local|crawled] [--top N] [--threshold 0.0-1.0][/red]")
                    else:
                        # Extract query (everything between quotes or after --search)
                        query_text = query.split("--search", 1)[1].strip()

                        # Extract source filter
                        source_filter = None
                        if "--source" in query_text:
                            source_part = query_text.split("--source")[1].strip().split()[0]
                            if source_part in ["local", "crawled"]:
                                source_filter = source_part
                                query_text = query_text.split("--source")[0].strip()

                        # Extract top_k
                        top_k = 10
                        if "--top" in query_text:
                            try:
                                top_k = int(query_text.split("--top")[1].strip().split()[0])
                                query_text = query_text.split("--top")[0].strip()
                            except (ValueError, IndexError):
                                pass

                        # Extract threshold
                        threshold = 0.5
                        if "--threshold" in query_text:
                            try:
                                threshold = float(query_text.split("--threshold")[1].strip().split()[0])
                                query_text = query_text.split("--threshold")[0].strip()
                            except (ValueError, IndexError):
                                pass

                        # Remove quotes if present
                        query_text = query_text.strip('"').strip("'").strip()

                        if query_text:
                            run_index_search(query_text, top_k=top_k, source_filter=source_filter, threshold=threshold)
                        else:
                            console.print("[red]Please provide a search query[/red]")
                else:
                    console.print("[yellow]Unknown index command. Try:[/yellow]")
                    console.print("[dim]  index --info     - Show index statistics[/dim]")
                    console.print("[dim]  index --rebuild  - Rebuild entire index[/dim]")
                    console.print("[dim]  index --clean    - Clean orphaned chunks[/dim]")
                    console.print("[dim]  index --search \"query\" [--threshold 0.3] - Search the index[/dim]")

                continue

            elif query.lower().startswith("crawl "):
                # Parse crawl command with options
                parts = query.split()

                if len(parts) < 2:
                    console.print("[red]Usage: crawl <url> | crawl --batch <file> | crawl --sitemap <url>[/red]")
                    continue

                # Check for batch mode
                if "--batch" in parts:
                    batch_idx = parts.index("--batch")
                    if batch_idx + 1 < len(parts):
                        file_path = parts[batch_idx + 1]

                        # Extract --parallel option
                        max_concurrent = 5
                        if "--parallel" in parts:
                            parallel_idx = parts.index("--parallel")
                            if parallel_idx + 1 < len(parts):
                                try:
                                    max_concurrent = int(parts[parallel_idx + 1])
                                except ValueError:
                                    pass

                        run_batch_crawl(file_path, max_concurrent=max_concurrent)
                    else:
                        console.print("[red]Usage: crawl --batch <file> [--parallel N][/red]")

                # Check for sitemap mode
                elif "--sitemap" in parts:
                    sitemap_idx = parts.index("--sitemap")
                    if sitemap_idx + 1 < len(parts):
                        sitemap_url = parts[sitemap_idx + 1]

                        # Extract --filter option
                        url_filter = None
                        if "--filter" in parts:
                            filter_idx = parts.index("--filter")
                            if filter_idx + 1 < len(parts):
                                url_filter = parts[filter_idx + 1]

                        # Extract --parallel option
                        max_concurrent = 5
                        if "--parallel" in parts:
                            parallel_idx = parts.index("--parallel")
                            if parallel_idx + 1 < len(parts):
                                try:
                                    max_concurrent = int(parts[parallel_idx + 1])
                                except ValueError:
                                    pass

                        run_sitemap_crawl(sitemap_url, url_filter=url_filter, max_concurrent=max_concurrent)
                    else:
                        console.print("[red]Usage: crawl --sitemap <url> [--filter pattern] [--parallel N][/red]")

                # Regular single URL crawl
                else:
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
