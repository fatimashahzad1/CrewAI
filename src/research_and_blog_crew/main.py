#!/usr/bin/env python
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from research_and_blog_crew.crew import ResearchAndBlogCrew

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def _topic_slug(topic: str) -> str:
    """Convert topic to a safe filename slug."""
    slug = topic.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug or "blog-post"


def _get_topic_from_api(source: str = "random") -> str:
    """Fetch a topic from the Topic API.
    
    Args:
        source: 'random' or 'trending' - determines which endpoint to use
        
    Returns:
        Topic title as string
        
    Raises:
        Exception: If API is unreachable or returns invalid data
    """
    api_url = os.getenv("TOPIC_API_URL", "http://localhost:3000")
    
    try:
        if source == "trending":
            response = requests.get(f"{api_url}/api/topics/trending", timeout=5)
            response.raise_for_status()
            data = response.json()
            topics = data.get("topics", [])
            if topics:
                # Return the first trending topic
                return topics[0].get("title", "AI Technology")
        else:
            # Default to random
            response = requests.get(f"{api_url}/api/topics/random", timeout=5)
            response.raise_for_status()
            data = response.json()
            topic = data.get("topic")
            if topic:
                return topic.get("title", "AI Technology")
        
        # Fallback if no topic found
        return "AI Technology"
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch topic from API ({e}). Using default topic.")
        return "AI Technology"


def run():
    """Run the crew; topic from CLI args, API, or fallback. Writes blog to blog/<slug>.md."""
    if len(sys.argv) > 1:
        # Allow manual topic override via CLI args
        topic = " ".join(sys.argv[1:]).strip()
        print(f"Using provided topic: {topic}")
    else:
        # Automatically fetch topic from API
        print("Fetching topic from API...")
        topic = _get_topic_from_api(source="random")
        print(f"Selected topic: {topic}")

    if not topic:
        print("Error: Could not determine topic.")
        sys.exit(1)

    inputs = {"topic": topic, "current_year": str(datetime.now().year)}

    try:
        result = ResearchAndBlogCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}") from e

    blog_dir = Path(os.getcwd()) / "blog"
    blog_dir.mkdir(parents=True, exist_ok=True)
    blog_path = blog_dir / f"{_topic_slug(topic)}.md"
    blog_content = result.raw or (
        result.tasks_output[-1].raw if result.tasks_output else ""
    )
    blog_path.write_text(blog_content, encoding="utf-8")
    print(f"\nBlog written to: {blog_path}")
    return result


def train():
    """Train the crew for a given number of iterations."""
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
    """Replay the crew execution from a specific task."""
    try:
        ResearchAndBlogCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def test():
    """Test the crew execution and return the results."""
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
    """Run the crew with trigger payload (JSON as first arg)."""
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
