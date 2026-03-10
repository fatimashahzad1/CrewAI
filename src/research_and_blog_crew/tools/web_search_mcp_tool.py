"""Tool for web search using DuckDuckGo (free, no API key required)."""

import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


class WebSearchInput(BaseModel):
    """Input schema for Web Search tool."""

    query: str = Field(
        description="Search query string to find information on the web"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of search results to return (default: 5, max: 20)",
    )


class WebSearchMCPTool(BaseTool):
    """Tool to search the web for current information using DuckDuckGo."""

    name: str = "web_search"
    description: str = (
        "Searches the web for current information, recent articles, and real-time data. "
        "Use this tool when you need: current/recent information about topics, "
        "verification of facts from web sources, recent articles and papers, "
        "real-world examples and applications, or latest developments. "
        "This tool provides up-to-date information from the internet that complements "
        "the structured topic data from the topic API."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def __init__(self):
        super().__init__()
        if DDGS is None:
            raise ImportError(
                "duckduckgo-search is required. Install it with: pip install duckduckgo-search"
            )

    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute web search and return formatted results."""
        try:
            if not query or not query.strip():
                return "Error: Search query cannot be empty."

            # Limit max_results to reasonable range
            max_results = min(max(1, max_results), 20)

            # Perform search using DuckDuckGo
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return f"No search results found for query: '{query}'"

            # Format results for agent consumption
            formatted_results = f"Web Search Results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"Result {i}:\n"
                formatted_results += f"Title: {result.get('title', 'N/A')}\n"
                formatted_results += f"URL: {result.get('href', 'N/A')}\n"
                formatted_results += f"Description: {result.get('body', 'N/A')}\n"
                formatted_results += "\n"

            return formatted_results

        except Exception as e:
            return f"Error performing web search: {str(e)}. Please try again with a different query."

