import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class ResearchAndBlogCrew:
    """Crew: research → report → review → blog → edit (5 agents, sequential)."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            llm=self.gemini_llm(),
            verbose=True,
        )

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generator"],  # type: ignore[index]
            llm=self.gemini_llm(),
            verbose=True,
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],  # type: ignore[index]
            llm=self.gemini_llm(),
            verbose=True,
        )

    @agent
    def blog_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["blog_writer"],  # type: ignore[index]
            llm=self.gemini_llm(),
            verbose=True,
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config["editor"],  # type: ignore[index]
            llm=self.gemini_llm(),
            verbose=True,
        )

    def gemini_llm(self) -> LLM:
        """LLM from env: MODEL and GEMINI_API_KEY."""
        model = os.getenv("MODEL", "gemini/gemini-2.5-flash")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return LLM(model=model, api_key=api_key)

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
