# Knowledge Graph

The Omnibus Legal Compass uses a Knowledge Graph to model the complex relationships between Indonesian legal entities.

## Schema

The graph is composed of several node types representing different levels of the legal hierarchy:

- **Law (UU)**: Undang-Undang
- **Government Regulation (PP)**: Peraturan Pemerintah
- **Presidential Regulation (Perpres)**: Peraturan Presiden
- **Ministerial Regulation (Permen)**: Peraturan Menteri
- **Chapter (Bab)**: Structural divisions within a document
- **Article (Pasal)**: The base unit of legal text

## Edge Types

Relationships between nodes define how the legal system is structured:

- **CONTAINS**: A document contains a chapter, or a chapter contains an article.
- **IMPLEMENTS**: A regulation (like a PP) provides the implementation details for a Law (UU).
- **AMENDS**: A newer document modifies parts of an existing one.
- **REFERENCES**: One document mentions another.
- **SUPERSEDES**: A newer document completely replaces an older one.

## ID Format

All nodes in the graph follow a standardized ID format for consistent retrieval:

`{jenis_dokumen}_{nomor}_{tahun}`

Examples:
- `uu_11_2020` (Undang-Undang Nomor 11 Tahun 2020)
- `pp_5_2021` (Peraturan Pemerintah Nomor 5 Tahun 2021)

## API Endpoints

Access the knowledge graph programmatically via these endpoints:

- `GET /api/v1/graph/laws`: List all indexed regulations.
- `GET /api/v1/graph/law/{id}`: Get details for a specific regulation.
- `GET /api/v1/graph/law/{id}/hierarchy`: Get the chapter/article hierarchy of a law.
- `GET /api/v1/graph/search`: Search for specific nodes in the graph.
- `GET /api/v1/graph/stats`: Get overview statistics of the graph.

## Data Ingestion

The Knowledge Graph is automatically populated during the ingestion process. The system parses legal documents stored in `data/peraturan/`, extracts structural elements (chapters, articles), and identifies cross-references to other laws to build the network of relationships.
