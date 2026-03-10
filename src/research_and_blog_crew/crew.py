import logging
import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task, tool
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import TavilySearchTool

from research_and_blog_crew.tools.topic_api_tool import TopicAPITool
from research_and_blog_crew.tools.web_search_mcp_tool import WebSearchMCPTool

logger = logging.getLogger(__name__)


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
        """Researcher tools: topic_api, web_search (DuckDuckGo), and optionally Tavily if TAVILY_API_KEY is set."""
        tools: list = [self.topic_api_tool(), self.web_search_mcp_tool()]
        tavily_enabled = bool(os.getenv("TAVILY_API_KEY", "").strip())
        if tavily_enabled:
            tools.append(_TavilySearchToolWithConsole())

        # Log so you can see which search backends are active (no MCP in use; Tavily = native tool)
        tool_names = ["topic_api", "web_search (DuckDuckGo)"]
        if tavily_enabled:
            tool_names.append("tavily_search (native Tavily API)")
        logger.info(
            "Researcher tools: %s. (MCP is not used; Tavily uses native TavilySearchTool when TAVILY_API_KEY is set.)",
            ", ".join(tool_names),
        )
        return tools

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=self._researcher_tools(),
            verbose=True,
        )

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generator"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=[self.topic_api_tool()],
            verbose=True,
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=[self.topic_api_tool()],
            verbose=True,
        )

    @agent
    def blog_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["blog_writer"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=[self.topic_api_tool()],
            verbose=True,
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config["editor"],  # type: ignore[index]
            llm=self.gemini_llm(),
            tools=[self.topic_api_tool()],
            verbose=True,
        )

    def gemini_llm(self) -> LLM:
        """LLM from env: MODEL and GEMINI_API_KEY."""
        model = os.getenv("MODEL", "gemini/gemini-3.1-flash-lite-preview")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return LLM(model=model, api_key=api_key)

    @tool
    def topic_api_tool(self) -> TopicAPITool:
        """Tool to interact with the Topic API."""
        return TopicAPITool()

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
