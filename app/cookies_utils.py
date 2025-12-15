import json
from typing import Any


def set_cookies(redis_client, name: str, value: Any):
    """Serialize `value` to JSON and store under `name`."""
    serialized_list = json.dumps(value)
    redis_client.set(name, serialized_list)


def get_cookies(redis_client, name: str):
    """Fetch JSON from Redis and deserialize, returning `None` if missing."""
    serialized_list = redis_client.get(name)
    if serialized_list:
        my_list = json.loads(serialized_list)
        return my_list
    else:
        return None


def clear_cookies(redis_client, name: str):
    """Delete the key `name` from Redis if present."""
    redis_client.delete(name)
"""Simple Redis-backed helpers to store and retrieve JSON serializable values.

Used for chat history storage keyed by session identifiers.
"""
