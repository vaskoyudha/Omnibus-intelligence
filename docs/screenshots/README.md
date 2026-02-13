# Screenshots

This directory contains screenshots of the Omnibus Legal Compass application for use in documentation and the README.

## Capturing Screenshots

To capture fresh screenshots:

1. Start the full application stack:
   ```bash
   # Terminal 1: Qdrant
   docker run -d --name omnibus-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

   # Terminal 2: Backend
   cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

   # Terminal 3: Frontend
   cd frontend && npm run dev
   ```

2. Navigate to each page and capture:
   - **Legal Q&A** (`/`) — Ask a question and capture the response with citations
   - **Compliance Checker** (`/compliance`) — Run a compliance check
   - **Business Guidance** (`/guidance`) — Generate guidance for a PT
   - **Multi-Turn Chat** (`/chat`) — Show a multi-turn conversation
   - **Knowledge Graph** (`/knowledge-graph`) — Expand the tree view
   - **Dashboard** (`/dashboard`) — Show heat map and bar charts

3. Save screenshots as PNG files in this directory:
   - `qa-page.png`
   - `compliance-page.png`
   - `guidance-page.png`
   - `chat-page.png`
   - `knowledge-graph-page.png`
   - `dashboard-page.png`

4. Recommended size: 1280x800 pixels
