# Features Overview

The Omnibus Legal Compass provides a suite of AI-powered tools designed to simplify the Indonesian regulatory landscape.

## Legal Q&A with Citations

The core feature of the system is the Legal Q&A. It allows users to ask natural language questions about Indonesian laws and receive detailed answers. Every answer is grounded in the retrieved legal context and includes precise citations to the source documents. This ensures that the AI does not hallucinate and provides verifiable information.
*   **Endpoint**: `POST /api/v1/ask`

## Compliance Checker

The Compliance Checker helps businesses determine if their operations align with current regulations. Users can provide a business description or upload PDF documents for analysis. The system identifies potential issues, suggests recommendations, and cites the relevant laws. This is particularly useful for auditing internal policies against national standards.
*   **Endpoint**: `POST /api/v1/compliance/check`

## Business Guidance

Setting up a business in Indonesia involves navigating complex bureaucratic steps. The Business Guidance feature provides a roadmap based on the business type, industry, and location. It lists required permits, estimated timelines, and issuing authorities. It also provides the legal basis for each step in the formation process.
*   **Endpoint**: `POST /api/v1/guidance`

## Multi-turn Chat

For more complex legal inquiries, the system supports multi-turn chat. This allows users to ask follow-up questions and refine their search within a single session. The system maintains context across the conversation, making it easier to explore specific legal topics in depth. Chat sessions are persisted for future reference.
*   **Endpoint**: `GET /api/v1/chat/sessions/{id}`

## Knowledge Graph

The Knowledge Graph visualizes the interconnected nature of Indonesian regulations. It displays relationships between different types of documents, such as how a Government Regulation (PP) implements a specific Law (UU). Users can explore the hierarchical structure of a law, including chapters and articles. This provides a structural understanding of the legal framework.
*   **Endpoint**: `GET /api/v1/graph/laws`

## Compliance Dashboard

The Compliance Dashboard offers a bird's-eye view of regulatory data. It includes a coverage heat map that shows which areas of law are most heavily documented in the system. It also provides statistics on the total number of regulations, articles, and relationships tracked. This dashboard helps administrators monitor the growth and health of the legal database.
*   **Endpoint**: `GET /api/v1/dashboard/stats`
