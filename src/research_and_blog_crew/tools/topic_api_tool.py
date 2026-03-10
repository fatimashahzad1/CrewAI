"""Tool for interacting with the Topic API from NestJS."""

import os
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TopicAPIInput(BaseModel):
    """Input schema for Topic API tool."""

    action: str = Field(
        description="Action to perform: 'get_all', 'get_by_id', 'get_trending', 'get_random', or 'search'"
    )
    topic_id: str = Field(
        default="", description="Topic ID (required for 'get_by_id' action)"
    )
    query: str = Field(
        default="", description="Search query (required for 'search' action)"
    )


class TopicAPITool(BaseTool):
    """Tool to fetch topics from the NestJS Topic API."""

    name: str = "topic_api"
    description: str = (
        "Fetches topic information from the Topic API. "
        "Use this to get available research topics, trending topics, or search for specific topics. "
        "Actions: 'get_all' - get all topics, 'get_by_id' - get topic by ID, "
        "'get_trending' - get trending topics, 'get_random' - get random topic, "
        "'search' - search topics by query."
    )
    args_schema: Type[BaseModel] = TopicAPIInput

    @property
    def _api_base(self) -> str:
        return os.getenv("TOPIC_API_URL", "http://localhost:3000")

    def _run(self, action: str, topic_id: str = "", query: str = "") -> str:
        """Execute the tool based on the action."""
        try:
            if action == "get_all":
                return self._get_all_topics()
            elif action == "get_by_id":
                if not topic_id:
                    return "Error: topic_id is required for 'get_by_id' action"
                return self._get_topic_by_id(topic_id)
            elif action == "get_trending":
                return self._get_trending_topics()
            elif action == "get_random":
                return self._get_random_topic()
            elif action == "search":
                if not query:
                    return "Error: query is required for 'search' action"
                return self._search_topics(query)
            else:
                return f"Error: Unknown action '{action}'. Valid actions: get_all, get_by_id, get_trending, get_random, search"
        except Exception as e:
            return f"Error calling Topic API: {str(e)}"

    def _get_all_topics(self) -> str:
        """Get all available topics."""
        response = requests.get(f"{self._api_base}/api/topics", timeout=10)
        response.raise_for_status()
        data = response.json()
        topics = data.get("topics", [])
        if not topics:
            return "No topics found."
        result = "Available Topics:\n\n"
        for topic in topics:
            result += f"ID: {topic['id']}\n"
            result += f"Title: {topic['title']}\n"
            result += f"Description: {topic['description']}\n"
            result += f"Category: {topic['category']}\n"
            result += f"Difficulty: {topic['difficulty']}\n"
            if topic.get("trending"):
                result += "Status: Trending\n"
            result += "\n"
        return result

    def _get_topic_by_id(self, topic_id: str) -> str:
        """Get a specific topic by ID."""
        response = requests.get(
            f"{self._api_base}/api/topics/{topic_id}", timeout=10
        )
        response.raise_for_status()
        data = response.json()
        topic = data.get("topic")
        if not topic:
            return f"Topic with ID {topic_id} not found."
        result = f"Topic Details:\n\n"
        result += f"ID: {topic['id']}\n"
        result += f"Title: {topic['title']}\n"
        result += f"Description: {topic['description']}\n"
        result += f"Category: {topic['category']}\n"
        result += f"Difficulty: {topic['difficulty']}\n"
        if topic.get("trending"):
            result += "Status: Trending\n"
        if topic.get("relatedTopics"):
            result += f"Related Topics: {', '.join(topic['relatedTopics'])}\n"
        return result

    def _get_trending_topics(self) -> str:
        """Get trending topics."""
        response = requests.get(
            f"{self._api_base}/api/topics/trending", timeout=10
        )
        response.raise_for_status()
        data = response.json()
        topics = data.get("topics", [])
        if not topics:
            return "No trending topics found."
        result = "Trending Topics:\n\n"
        for topic in topics:
            result += f"ID: {topic['id']}\n"
            result += f"Title: {topic['title']}\n"
            result += f"Description: {topic['description']}\n"
            result += f"Category: {topic['category']}\n"
            result += "\n"
        return result

    def _get_random_topic(self) -> str:
        """Get a random topic."""
        response = requests.get(
            f"{self._api_base}/api/topics/random", timeout=10
        )
        response.raise_for_status()
        data = response.json()
        topic = data.get("topic")
        if not topic:
            return "No topic found."
        result = "Random Topic:\n\n"
        result += f"ID: {topic['id']}\n"
        result += f"Title: {topic['title']}\n"
        result += f"Description: {topic['description']}\n"
        result += f"Category: {topic['category']}\n"
        result += f"Difficulty: {topic['difficulty']}\n"
        if topic.get("trending"):
            result += "Status: Trending\n"
        return result

    def _search_topics(self, query: str) -> str:
        """Search topics by query."""
        response = requests.get(
            f"{self._api_base}/api/topics/search",
            params={"q": query},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        topics = data.get("topics", [])
        if not topics:
            return f"No topics found matching query: '{query}'"
        result = f"Search Results for '{query}':\n\n"
        for topic in topics:
            result += f"ID: {topic['id']}\n"
            result += f"Title: {topic['title']}\n"
            result += f"Description: {topic['description']}\n"
            result += f"Category: {topic['category']}\n"
            result += "\n"
        return result
