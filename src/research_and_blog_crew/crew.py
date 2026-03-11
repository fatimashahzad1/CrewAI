import logging
import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task, tool
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.mcp import MCPServerStdio
from crewai_tools import TavilySearchTool

from research_and_blog_crew.tools.web_search_mcp_tool import WebSearchMCPTool

logger = logging.getLogger(__name__)


def _topic_mcp_stdio() -> MCPServerStdio:
    """Build MCPServerStdio for the NestJS topic-api (stdio). Path is topic-api/dist/main.js relative to repo root (sibling of research_and_blog_crew)."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "topic-api"))
    script = os.path.join(base, "dist", "main.js")
    return MCPServerStdio(command="node", args=[script])


def _topic_mcps() -> list:
    """MCP config list for the Topic API server (stdio)."""
    return [_topic_mcp_stdio()]


class _TavilySearchToolWithConsole(TavilySearchTool):
    """TavilySearchTool that prints its result to the console when used."""

    def _run(self, *args, **kwargs):
        result = super()._run(*args, **kwargs)
        print("\n--- Tavily search output ---")
        print(result if isinstance(result, str) else str(result))
        print("---\n")
        return result


@CrewBase
class ResearchAndBlogCrew:
    """Crew: research → report → review → blog → edit (5 agents, sequential)."""

    agents: list[BaseAgent]
    tasks: list[Task]

    def _researcher_tools(self) -> list:
        """Researcher tools: Topic MCP (get_all_topics, get_topic_by_id, etc.), web_search (DuckDuckGo), optionally Tavily."""
        tools: list = [self.web_search_mcp_tool()]
        tavily_enabled = bool(os.getenv("TAVILY_API_KEY", "").strip())
        if tavily_enabled:
            tools.append(_TavilySearchToolWithConsole())
        tool_names = ["topic MCP (get_all_topics, get_topic_by_id, get_trending_topics, get_random_topic, search_topics)", "web_search (DuckDuckGo)"]
        if tavily_enabled:
            tool_names.append("tavily_search (Tavily API)")
        logger.info("Researcher tools: %s.", ", ".join(tool_names))
        return tools

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=self._researcher_tools(),
            mcps=_topic_mcps(),
            verbose=True,
        )

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generator"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=[],
            mcps=_topic_mcps(),
            verbose=True,
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=[],
            mcps=_topic_mcps(),
            verbose=True,
        )

    @agent
    def blog_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["blog_writer"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=[],
            mcps=_topic_mcps(),
            verbose=True,
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config["editor"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=[],
            mcps=_topic_mcps(),
            verbose=True,
        )

    def gemini_llm(self) -> LLM:
        """LLM from env: MODEL and GEMINI_API_KEY."""
        model = os.getenv("MODEL", "gemini/gemini-3.1-flash-lite-preview")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return LLM(model=model, api_key=api_key)

    @tool
    def web_search_mcp_tool(self) -> WebSearchMCPTool:
        """Tool to search the web for current information."""
        return WebSearchMCPTool()

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def report_task(self) -> Task:
        return Task(
            config=self.tasks_config["report_task"],  # type: ignore[index]
        )

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_task"],  # type: ignore[index]
        )

    @task
    def blog_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["blog_writing_task"],  # type: ignore[index]
        )

    @task
    def edit_task(self) -> Task:
        return Task(
            config=self.tasks_config["edit_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Sequential crew: research → report → review → blog → edit."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
