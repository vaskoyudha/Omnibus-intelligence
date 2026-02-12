"""
In-memory chat session manager with auto-expiry.

Provides session creation, message storage with a sliding window,
and automatic cleanup of inactive sessions.
"""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = Field(default_factory=time.time)


class Session(BaseModel):
    """A chat session with message history."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = []
    created_at: float = Field(default_factory=time.time)
    last_active: float = Field(default_factory=time.time)


class SessionManager:
    """In-memory chat session storage with sliding window and auto-expiry.

    Parameters
    ----------
    max_messages:
        Maximum number of messages retained per session (sliding window).
    expiry_seconds:
        Seconds of inactivity after which a session is considered expired.
    """

    def __init__(
        self,
        max_messages: int = 10,
        expiry_seconds: float = 1800,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self.max_messages = max_messages
        self.expiry_seconds = expiry_seconds

    def create_session(self) -> str:
        """Create a new session and return its UUID."""
        session = Session()
        self._sessions[session.id] = session
        return session.id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session, trimming to the sliding window."""
        session = self._get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found or expired")
        session.messages.append(Message(role=role, content=content))
        if len(session.messages) > self.max_messages:
            session.messages = session.messages[-self.max_messages :]
        session.last_active = time.time()

    def get_history(self, session_id: str) -> list[Message]:
        """Return all messages in the session, or empty list if not found."""
        session = self._get_session(session_id)
        if session is None:
            return []
        return session.messages

    def get_chat_history_for_rag(self, session_id: str) -> list[dict[str, str]]:
        """Convert session history to ``[{question, answer}, ...]`` format
        expected by :meth:`LegalRAGChain.query_with_history`.
        """
        messages = self.get_history(session_id)
        history: list[dict[str, str]] = []
        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                history.append(
                    {
                        "question": messages[i].content,
                        "answer": messages[i + 1].content,
                    }
                )
                i += 2
            else:
                i += 1
        return history

    def clear_session(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > self.expiry_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def _get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.last_active > self.expiry_seconds:
            del self._sessions[session_id]
            return None
        return session
