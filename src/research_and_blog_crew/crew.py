import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class ResearchAndBlogCrew:
    """Crew: research an AI topic → report → blog post."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generator"],  # type: ignore[index]
            llm=self.openai_llm(),
            verbose=True,
        )

    @agent
    def blog_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["blog_writer"],  # type: ignore[index]
            llm=self.openai_llm(),
            verbose=True,
        )

    def openai_llm(self) -> LLM:
        """LLM from env: MODEL and OPENAI_API_KEY."""
        model = os.getenv("MODEL", "gpt-4")
        return LLM(model=model)

    @task
    def report_task(self) -> Task:
        return Task(
            config=self.tasks_config["report_task"],  # type: ignore[index]
        )

    @task
    def blog_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["blog_writing_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ResearchAndBlogCrew with sequential flow: report → blog."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
