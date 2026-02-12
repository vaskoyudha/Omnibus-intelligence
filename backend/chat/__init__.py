"""
Chat module for multi-turn conversation support.

Provides in-memory session management with sliding window context
for use with the RAG chain's ``query_with_history`` method.
"""

from .session import Message, Session, SessionManager

__all__ = ["Message", "Session", "SessionManager"]
