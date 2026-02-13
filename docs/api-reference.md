# API Reference

The Omnibus Legal Compass provides a RESTful API for interacting with the RAG system and Knowledge Graph.

## Base URL
```
http://localhost:8000
```

## Versioning
- `/api/v1/`: Current stable version (Recommended)
- `/api/`: Legacy endpoints

## Rate Limiting
To ensure stability, the API is rate-limited to **30 requests per minute** per IP address across all endpoints.

---

## Endpoints

### Legal Q&A
`POST /api/v1/ask`

Ask a legal question and get a response with citations.

**Request Body:**
```json
{
  "question": "Apa syarat pendirian PT?"
}
```

**Response:**
```json
{
  "answer": "Syarat pendirian PT menurut UU 40/2007 adalah...",
  "citations": [
    {
      "source": "UU 40/2007",
      "excerpt": "Perseroan didirikan oleh 2 orang atau lebih...",
      "relevance_score": 0.95
    }
  ]
}
```

### Streaming Q&A
`POST /api/v1/ask/stream`

Get answers in real-time via Server-Sent Events (SSE).

### Compliance Checker
`POST /api/v1/compliance/check`

Check business description or PDF against regulations. Supports `multipart/form-data` for file uploads.

### Business Guidance
`POST /api/v1/guidance`

Get guidance on business formation.

**Request Body:**
```json
{
  "business_type": "PT",
  "industry": "IT",
  "location": "Jakarta"
}
```

### Chat History
`GET /api/v1/chat/sessions/{id}`

Retrieve messages from a specific chat session.

### Knowledge Graph

- `GET /api/v1/graph/laws`: List all indexed regulations.
- `GET /api/v1/graph/law/{id}/hierarchy`: Get the hierarchical structure (Chapters/Articles).
- `GET /api/v1/graph/search?q=...`: Search the knowledge graph.

### Dashboard Data

- `GET /api/v1/dashboard/coverage`: Get regulatory coverage data for the heat map.
- `GET /api/v1/dashboard/stats`: Get system-wide statistics.

### Health Check
`GET /health`

Returns the current status of the API and database connections.

---

## Error Format

All errors return a standard JSON object:

```json
{
  "detail": "Error message description"
}
```
