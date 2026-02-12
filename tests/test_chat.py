"""
Tests for multi-turn chat session management.

Covers:
- SessionManager unit tests (create, message, history, sliding window, expiry)
- API endpoint integration tests (POST /ask with session_id, GET/DELETE /chat/sessions)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from chat.session import Message, Session, SessionManager
from rag_chain import RAGResponse, ConfidenceScore, ValidationResult


# ---------------------------------------------------------------------------
# Helper — build a complete RAGResponse for mocking query() / query_with_history()
# ---------------------------------------------------------------------------


def _mock_rag_response(answer: str = "Jawaban [1].", n_citations: int = 2) -> RAGResponse:
    citations = [
        {
            "number": i,
            "citation_id": f"uu-{i}-2020",
            "citation": f"UU No. {i} Tahun 2020, Pasal {i}",
            "score": 0.85 - (i * 0.05),
            "metadata": {"jenis_dokumen": "UU", "text": "..."},
        }
        for i in range(1, n_citations + 1)
    ]
    return RAGResponse(
        answer=answer,
        citations=citations,
        sources=[f"[{c['number']}] {c['citation']}" for c in citations],
        confidence="tinggi",
        confidence_score=ConfidenceScore(
            numeric=0.82, label="tinggi", top_score=0.9, avg_score=0.75
        ),
        raw_context="context",
        validation=ValidationResult(
            is_valid=True, citation_coverage=0.8, hallucination_risk="low"
        ),
    )


# ===========================================================================
# SessionManager unit tests
# ===========================================================================


class TestSessionManagerCreate:
    def test_create_session_returns_uuid(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID4 format

    def test_create_multiple_sessions(self):
        sm = SessionManager()
        ids = {sm.create_session() for _ in range(5)}
        assert len(ids) == 5  # All unique


class TestSessionManagerMessages:
    def test_add_and_get_history(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, "user", "Hello")
        sm.add_message(sid, "assistant", "Hi there")

        history = sm.get_history(sid)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there"

    def test_add_message_to_nonexistent_session_raises(self):
        sm = SessionManager()
        with pytest.raises(KeyError, match="not found"):
            sm.add_message("nonexistent-id", "user", "Hello")

    def test_get_history_nonexistent_returns_empty(self):
        sm = SessionManager()
        assert sm.get_history("nonexistent-id") == []

    def test_message_has_timestamp(self):
        sm = SessionManager()
        sid = sm.create_session()
        before = time.time()
        sm.add_message(sid, "user", "Test")
        after = time.time()

        msg = sm.get_history(sid)[0]
        assert before <= msg.timestamp <= after


class TestSessionManagerSlidingWindow:
    def test_sliding_window_trims_oldest(self):
        sm = SessionManager(max_messages=4)
        sid = sm.create_session()
        for i in range(6):
            sm.add_message(sid, "user", f"msg-{i}")

        history = sm.get_history(sid)
        assert len(history) == 4
        # Oldest messages should have been trimmed
        assert history[0].content == "msg-2"
        assert history[-1].content == "msg-5"

    def test_sliding_window_default_10(self):
        sm = SessionManager()
        sid = sm.create_session()
        for i in range(15):
            sm.add_message(sid, "user", f"msg-{i}")

        history = sm.get_history(sid)
        assert len(history) == 10


class TestSessionManagerChatHistoryForRAG:
    def test_pairs_user_assistant_messages(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, "user", "What is PT?")
        sm.add_message(sid, "assistant", "PT is a limited liability company")
        sm.add_message(sid, "user", "What about CV?")
        sm.add_message(sid, "assistant", "CV is a limited partnership")

        rag_history = sm.get_chat_history_for_rag(sid)
        assert len(rag_history) == 2
        assert rag_history[0] == {
            "question": "What is PT?",
            "answer": "PT is a limited liability company",
        }
        assert rag_history[1] == {
            "question": "What about CV?",
            "answer": "CV is a limited partnership",
        }

    def test_empty_session_returns_empty_history(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert sm.get_chat_history_for_rag(sid) == []

    def test_nonexistent_session_returns_empty(self):
        sm = SessionManager()
        assert sm.get_chat_history_for_rag("nonexistent") == []

    def test_unpaired_trailing_user_message_excluded(self):
        """A trailing user message without an assistant reply should not appear."""
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, "user", "Q1")
        sm.add_message(sid, "assistant", "A1")
        sm.add_message(sid, "user", "Q2")  # no assistant reply yet

        rag_history = sm.get_chat_history_for_rag(sid)
        assert len(rag_history) == 1


class TestSessionManagerClear:
    def test_clear_existing_session(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert sm.clear_session(sid) is True
        assert sm.get_history(sid) == []

    def test_clear_nonexistent_session(self):
        sm = SessionManager()
        assert sm.clear_session("nonexistent") is False


class TestSessionManagerExpiry:
    def test_expired_session_is_invisible(self):
        sm = SessionManager(expiry_seconds=0.01)
        sid = sm.create_session()
        sm.add_message(sid, "user", "Hello")
        time.sleep(0.02)
        # Session should be expired
        assert sm.get_history(sid) == []

    def test_cleanup_expired_removes_old_sessions(self):
        sm = SessionManager(expiry_seconds=0.01)
        sid1 = sm.create_session()
        sid2 = sm.create_session()
        time.sleep(0.02)
        removed = sm.cleanup_expired()
        assert removed == 2

    def test_active_session_not_expired(self):
        sm = SessionManager(expiry_seconds=10)
        sid = sm.create_session()
        sm.add_message(sid, "user", "Hello")
        assert len(sm.get_history(sid)) == 1

    def test_add_message_to_expired_session_raises(self):
        sm = SessionManager(expiry_seconds=0.01)
        sid = sm.create_session()
        time.sleep(0.02)
        with pytest.raises(KeyError, match="not found"):
            sm.add_message(sid, "user", "Hello")


# ===========================================================================
# Pydantic model tests
# ===========================================================================


class TestMessageModel:
    def test_message_fields(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, float)


class TestSessionModel:
    def test_session_defaults(self):
        s = Session()
        assert len(s.id) == 36
        assert s.messages == []
        assert isinstance(s.created_at, float)
        assert isinstance(s.last_active, float)


# ===========================================================================
# API endpoint integration tests
# ===========================================================================


class TestAskWithSession:
    """Tests for POST /api/ask with session_id integration."""

    def test_ask_creates_session_when_no_session_id(self, test_client):
        """First call without session_id should create one and return it."""
        with patch("main.rag_chain") as mock_chain:
            mock_chain.query.return_value = _mock_rag_response()
            mock_chain.llm_client = MagicMock()
            mock_chain.retriever = MagicMock()
            mock_chain.retriever.client = MagicMock()

            response = test_client.post(
                "/api/ask",
                json={"question": "Apa itu Undang-Undang Cipta Kerja?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] is not None
        assert len(data["session_id"]) == 36  # UUID4

    def test_ask_with_session_id_uses_history(self, test_client):
        """Passing an existing session_id should use query_with_history when history exists."""
        with patch("main.rag_chain") as mock_chain, \
             patch("main.session_manager") as mock_sm:
            mock_chain.query.return_value = _mock_rag_response()
            mock_chain.query_with_history.return_value = _mock_rag_response(
                answer="Follow-up answer [1]."
            )
            mock_chain.llm_client = MagicMock()
            mock_chain.retriever = MagicMock()
            mock_chain.retriever.client = MagicMock()

            # Simulate existing session with history
            mock_sm.get_chat_history_for_rag.return_value = [
                {"question": "Q1", "answer": "A1"}
            ]
            mock_sm.create_session.return_value = "test-session-id"

            response = test_client.post(
                "/api/ask",
                json={
                    "question": "Bagaimana sanksinya?",
                    "session_id": "existing-session-id",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-id"
        # Should have used query_with_history, not query
        mock_chain.query_with_history.assert_called_once()
        mock_chain.query.assert_not_called()

    def test_ask_with_session_id_no_history_uses_query(self, test_client):
        """Session exists but has no history yet — should use regular query."""
        with patch("main.rag_chain") as mock_chain, \
             patch("main.session_manager") as mock_sm:
            mock_chain.query.return_value = _mock_rag_response()
            mock_chain.llm_client = MagicMock()
            mock_chain.retriever = MagicMock()
            mock_chain.retriever.client = MagicMock()

            # Session exists but empty history (fresh session)
            mock_sm.get_chat_history_for_rag.return_value = []

            response = test_client.post(
                "/api/ask",
                json={
                    "question": "Apa itu PT?",
                    "session_id": "fresh-session-id",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "fresh-session-id"
        mock_chain.query.assert_called_once()
        mock_chain.query_with_history.assert_not_called()

    def test_ask_records_messages_in_session(self, test_client):
        """After a successful ask, user + assistant messages should be recorded."""
        with patch("main.rag_chain") as mock_chain, \
             patch("main.session_manager") as mock_sm:
            mock_chain.query.return_value = _mock_rag_response(answer="My answer")
            mock_chain.llm_client = MagicMock()
            mock_chain.retriever = MagicMock()
            mock_chain.retriever.client = MagicMock()

            mock_sm.get_chat_history_for_rag.return_value = []
            mock_sm.create_session.return_value = "new-sid"

            response = test_client.post(
                "/api/ask",
                json={"question": "Test question"},
            )

        assert response.status_code == 200
        # Should have recorded user question and assistant answer
        calls = mock_sm.add_message.call_args_list
        assert len(calls) == 2
        assert calls[0].args == ("new-sid", "user", "Test question")
        assert calls[1].args == ("new-sid", "assistant", "My answer")


class TestChatSessionEndpoints:
    """Tests for GET and DELETE /api/chat/sessions/{session_id}."""

    def test_get_session_history(self, test_client):
        """GET /api/chat/sessions/{id} returns message history."""
        with patch("main.session_manager") as mock_sm:
            mock_sm.get_history.return_value = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ]

            response = test_client.get("/api/chat/sessions/test-sid")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-sid"
        assert len(data["messages"]) == 2
        assert data["messages"][0] == {"role": "user", "content": "Hello"}
        assert data["messages"][1] == {"role": "assistant", "content": "Hi there"}

    def test_get_session_not_found(self, test_client):
        """GET /api/chat/sessions/{id} with nonexistent → 404."""
        with patch("main.session_manager") as mock_sm:
            mock_sm.get_history.return_value = []

            response = test_client.get("/api/chat/sessions/nonexistent")

        assert response.status_code == 404

    def test_delete_session(self, test_client):
        """DELETE /api/chat/sessions/{id} removes session."""
        with patch("main.session_manager") as mock_sm:
            mock_sm.clear_session.return_value = True

            response = test_client.delete("/api/chat/sessions/test-sid")

        assert response.status_code == 200
        data = response.json()
        assert data["detail"] == "Session deleted"
        assert data["session_id"] == "test-sid"

    def test_delete_session_not_found(self, test_client):
        """DELETE /api/chat/sessions/{id} with nonexistent → 404."""
        with patch("main.session_manager") as mock_sm:
            mock_sm.clear_session.return_value = False

            response = test_client.delete("/api/chat/sessions/nonexistent")

        assert response.status_code == 404

    def test_get_session_v1_prefix(self, test_client):
        """GET /api/v1/chat/sessions/{id} should also work."""
        with patch("main.session_manager") as mock_sm:
            mock_sm.get_history.return_value = [
                Message(role="user", content="Test"),
            ]

            response = test_client.get("/api/v1/chat/sessions/test-sid")

        assert response.status_code == 200

    def test_delete_session_v1_prefix(self, test_client):
        """DELETE /api/v1/chat/sessions/{id} should also work."""
        with patch("main.session_manager") as mock_sm:
            mock_sm.clear_session.return_value = True

            response = test_client.delete("/api/v1/chat/sessions/test-sid")

        assert response.status_code == 200


class TestAskSessionE2E:
    """End-to-end test: create session via /ask, retrieve via /chat/sessions."""

    def test_multi_turn_flow(self, test_client):
        """Full multi-turn conversation flow without mocking session_manager."""
        # Step 1: First question (no session_id) — creates a session
        with patch("main.rag_chain") as mock_chain:
            mock_chain.query.return_value = _mock_rag_response(answer="Answer 1")
            mock_chain.llm_client = MagicMock()
            mock_chain.retriever = MagicMock()
            mock_chain.retriever.client = MagicMock()

            resp1 = test_client.post(
                "/api/ask",
                json={"question": "Apa itu PT?"},
            )

        assert resp1.status_code == 200
        sid = resp1.json()["session_id"]
        assert sid is not None

        # Step 2: Retrieve session — should have 2 messages (user + assistant)
        resp2 = test_client.get(f"/api/chat/sessions/{sid}")
        assert resp2.status_code == 200
        messages = resp2.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Apa itu PT?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Answer 1"

        # Step 3: Follow-up question with session_id
        with patch("main.rag_chain") as mock_chain:
            mock_chain.query_with_history.return_value = _mock_rag_response(
                answer="Answer 2"
            )
            mock_chain.llm_client = MagicMock()
            mock_chain.retriever = MagicMock()
            mock_chain.retriever.client = MagicMock()

            resp3 = test_client.post(
                "/api/ask",
                json={"question": "Bagaimana cara mendirikannya?", "session_id": sid},
            )

        assert resp3.status_code == 200
        assert resp3.json()["session_id"] == sid

        # Step 4: Retrieve session — should now have 4 messages
        resp4 = test_client.get(f"/api/chat/sessions/{sid}")
        assert resp4.status_code == 200
        messages = resp4.json()["messages"]
        assert len(messages) == 4

        # Step 5: Delete session
        resp5 = test_client.delete(f"/api/chat/sessions/{sid}")
        assert resp5.status_code == 200

        # Step 6: Session should be gone
        resp6 = test_client.get(f"/api/chat/sessions/{sid}")
        assert resp6.status_code == 404
