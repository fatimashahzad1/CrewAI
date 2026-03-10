#!/usr/bin/env python
import os
import re
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from datetime import datetime

from research_and_blog_crew.crew import ResearchAndBlogCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Load MODEL and OPENAI_API_KEY from .env
load_dotenv()


def _topic_slug(topic: str) -> str:
    """Convert topic to a safe filename slug."""
    slug = topic.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug or "blog-post"


def run():
    """
    Run the crew. Topic must be an AI-related topic (e.g. given as first CLI arg or prompt).
    Writes the final blog to blog/<topic-slug>.md in the project root.
    """
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:]).strip()
    else:
        topic = input("Enter an AI-related topic for the report and blog: ").strip()

    if not topic:
        print("Error: Topic is required. Pass it as an argument or enter when prompted.")
        sys.exit(1)

    inputs = {
        "topic": topic,
        "current_year": str(datetime.now().year),
    }

    try:
        crew_instance = ResearchAndBlogCrew().crew()
        result = crew_instance.kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}") from e

    # Save blog output to blog/<topic-slug>.md (project root = cwd when run from root)
    blog_dir = Path(os.getcwd()) / "blog"
    blog_dir.mkdir(parents=True, exist_ok=True)
    slug = _topic_slug(topic)
    blog_path = blog_dir / f"{slug}.md"

    blog_content = result.raw if result else ""
    if not blog_content and result and result.tasks_output:
        blog_content = result.tasks_output[-1].raw if result.tasks_output else ""

    blog_path.write_text(blog_content, encoding="utf-8")
    print(f"\nBlog written to: {blog_path}")

    return result


def train():
    """
    Train the crew for a given number of iterations.
    """
    topic = sys.argv[2] if len(sys.argv) > 2 else "AI LLMs"
    inputs = {
        "topic": topic,
        "current_year": str(datetime.now().year),
    }
    try:
        ResearchAndBlogCrew().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}") from e


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ResearchAndBlogCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def test():
    """
    Test the crew execution and returns the results.
    """
    topic = sys.argv[3] if len(sys.argv) > 3 else "AI LLMs"
    inputs = {
        "topic": topic,
        "current_year": str(datetime.now().year),
    }
    try:
        ResearchAndBlogCrew().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}") from e


def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Please provide JSON payload as argument."
        )

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": "",
    }

    try:
        result = ResearchAndBlogCrew().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(
            f"An error occurred while running the crew with trigger: {e}"
        ) from e
