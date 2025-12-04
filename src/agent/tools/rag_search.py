"""
RAG Search Tool for the AI Coding Agent.

This tool provides a semantic search interface over the codebase for the agent,
using the underlying RAG system.
"""

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from src.rag.retriever import get_retriever, IndexNotFoundError
from src.config.settings import Settings

class RagSearchInput(BaseModel):
    """Input schema for the RAG Search tool."""
    query: str = Field(description="The natural language query for semantic codebase search.")

class RagSearchTool(BaseTool):

    """

    Tool to perform semantic search over the codebase using the RAG system.

    """

    name: str = "rag_search"

    description: str = (

        "Searches the codebase for relevant code snippets, documentation, and files "

        "based on a natural language query. Use this to understand project architecture, "

        "find function definitions, or locate relevant examples."

    )

    args_schema: type[BaseModel] = RagSearchInput

    settings: Settings



    def _run(self, query: str) -> str:

        """

        Execute the RAG search and format the results for the LLM.



        Args:

            query: The natural language query.



        Returns:

            A formatted string of search results or an error message.

        """

        try:

            retriever = get_retriever(settings=self.settings)

            results = retriever.search(query, top_k=5)



            if not results:

                return f"No relevant code snippets found for the query: '{query}'"



            formatted_results = [

                f"Found {len(results)} relevant code snippets for your query '{query}':\n"

            ]

            for i, result in enumerate(results, 1):

                snippet = (

                    f'{i}. File: {result["file_path"]} '

                    f'(lines {result["start_line"]}-{result["end_line"]}), '

                    f'Score: {result["score"]:.2f}\n'

                    f'---\n'

                    f'{result["content"]}\n'

                    f'---'

                )

                formatted_results.append(snippet)

            

            return "\n\n".join(formatted_results)



        except IndexNotFoundError:

            return (

                "Error: The codebase index has not been built yet. "

                "Please ask the user to run the indexing command first."

            )

        except Exception as e:

            return f"An unexpected error occurred during RAG search: {e}"



    async def _arun(self, query: str) -> str:

        """Async version (not implemented, falls back to sync)."""

        return self._run(query)



def get_rag_search_tool(settings: Settings) -> RagSearchTool:

    """

    Factory function to get an instance of the RagSearchTool.

    

    Args:

        settings: The application settings object.

        

    Returns:

        An instance of RagSearchTool.

    """

    return RagSearchTool(settings=settings)
