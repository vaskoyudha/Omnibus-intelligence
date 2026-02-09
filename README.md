# Omnibus Legal Compass 🧭

**Indonesian Legal RAG System** - Sistem Tanya Jawab Hukum Indonesia dengan AI

A production-ready Retrieval-Augmented Generation (RAG) system for Indonesian legal documents, powered by NVIDIA NIM (Kimi K2 model) and Qdrant vector database.

## ✨ Features

### 1. Legal Q&A with Citations (Tanya Jawab Hukum)
Ask questions about Indonesian regulations and receive accurate answers with source citations.

### 2. Compliance Checker (Pemeriksaan Kepatuhan)
Check if your business operations comply with Indonesian regulations. Supports both text input and PDF document analysis.

### 3. Business Formation Guidance (Panduan Pendirian Usaha)
Get step-by-step guidance on establishing a business in Indonesia, including required permits and regulatory steps.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend                         │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐    │
│  │   Q&A Page  │  │ Compliance Page │  │  Guidance Page   │    │
│  └─────────────┘  └─────────────────┘  └──────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/REST
┌───────────────────────────▼─────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐    │
│  │  /api/ask   │  │/api/compliance  │  │  /api/guidance   │    │
│  └──────┬──────┘  └────────┬────────┘  └────────┬─────────┘    │
│         └──────────────────┼────────────────────┘              │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    RAG Chain                             │   │
│  │  ┌────────────────┐    ┌─────────────────────────────┐  │   │
│  │  │ Hybrid Search  │───▶│      NVIDIA NIM (Kimi K2)   │  │   │
│  │  │ BM25 + Dense   │    │  moonshotai/kimi-k2-instruct│  │   │
│  │  └───────┬────────┘    └─────────────────────────────┘  │   │
│  └──────────┼──────────────────────────────────────────────┘   │
└─────────────┼───────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                      Qdrant Vector DB                            │
│  Collection: indonesian_legal_docs                               │
│  Embeddings: paraphrase-multilingual-MiniLM-L12-v2 (384 dim)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

- **Docker** (for Qdrant) - [Install Docker](https://docs.docker.com/get-docker/)
- **Python 3.11+** - [Install Python](https://www.python.org/downloads/)
- **Node.js 18+** - [Install Node.js](https://nodejs.org/)
- **NVIDIA NIM API Key** (Free tier available) - [Get API Key](https://build.nvidia.com/)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/regulatory-harmonization-engine.git
cd "Regulatory Harmonization Engine"
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
QDRANT_URL=http://localhost:6333
```

### 3. Start Qdrant Database

```bash
docker run -d --name omnibus-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

### 4. Set Up Python Backend

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Ingest Legal Documents

```bash
cd backend
python scripts/ingest.py
```

You should see: `✅ Successfully ingested 10 documents into Qdrant`

### 6. Start Backend Server

```bash
# From project root
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Set Up Frontend

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

### 8. Access the Application

Open your browser and navigate to:
- **Main App**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Legal Q&A - `POST /api/ask`

Ask questions about Indonesian legal regulations.

**Request:**
```json
{
  "question": "Apa syarat pendirian PT?"
}
```

**Response:**
```json
{
  "answer": "Berdasarkan peraturan yang berlaku, syarat pendirian PT meliputi...",
  "citations": [
    {
      "source": "UU 40/2007 tentang Perseroan Terbatas",
      "excerpt": "Perseroan didirikan oleh 2 (dua) orang atau lebih...",
      "relevance_score": 0.85
    }
  ],
  "confidence": 0.92
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Apa syarat pendirian PT?"}'
```

---

#### 2. Compliance Check - `POST /api/compliance/check`

Check business compliance against Indonesian regulations.

**Request (Text):**
```json
{
  "business_description": "Perusahaan ekspor impor dengan NIB"
}
```

**Request (PDF):**
```
Content-Type: multipart/form-data
file: [PDF file]
```

**Response:**
```json
{
  "compliance_status": "partially_compliant",
  "issues": [
    "Izin ekspor belum terdaftar"
  ],
  "recommendations": [
    "Daftarkan izin ekspor melalui OSS"
  ],
  "citations": [
    {
      "source": "PP 5/2021",
      "excerpt": "..."
    }
  ]
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/compliance/check \
  -H "Content-Type: application/json" \
  -d '{"business_description": "Perusahaan perdagangan umum dengan SIUP"}'
```

---

#### 3. Business Guidance - `POST /api/guidance`

Get step-by-step guidance for business formation.

**Request:**
```json
{
  "business_type": "PT",
  "industry": "teknologi",
  "location": "Jakarta"
}
```

**Response:**
```json
{
  "business_type": "PT",
  "business_type_name": "Perseroan Terbatas",
  "summary": "Panduan lengkap pendirian PT di bidang teknologi...",
  "steps": [
    {
      "step_number": 1,
      "title": "Pemesanan Nama PT",
      "description": "Ajukan pemesanan nama melalui AHU Online...",
      "estimated_time": "1-3 hari kerja",
      "requirements": ["KTP pendiri", "NPWP"]
    }
  ],
  "required_permits": [
    {
      "permit_name": "NIB",
      "issuing_authority": "OSS",
      "estimated_time": "1 hari"
    }
  ],
  "citations": [...]
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/guidance \
  -H "Content-Type: application/json" \
  -d '{"business_type": "CV", "industry": "perdagangan", "location": "Surabaya"}'
```

---

## 🗂️ Project Structure

```
Regulatory Harmonization Engine/
├── backend/
│   ├── main.py              # FastAPI application & endpoints
│   ├── rag_chain.py         # RAG chain with NVIDIA NIM
│   ├── retriever.py         # Hybrid search (BM25 + dense)
│   ├── scripts/
│   │   └── ingest.py        # Document ingestion script
│   └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx           # Q&A page
│   │   │   ├── compliance/
│   │   │   │   └── page.tsx       # Compliance checker
│   │   │   ├── guidance/
│   │   │   │   └── page.tsx       # Business guidance
│   │   │   └── layout.tsx         # App layout with navigation
│   │   ├── components/
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── CitationCard.tsx
│   │   │   ├── LoadingSpinner.tsx
│   │   │   └── QuestionInput.tsx
│   │   └── lib/
│   │       └── api.ts             # API client
│   ├── package.json
│   └── .env.local
├── data/
│   └── peraturan/
│       └── sample.json            # Sample legal documents
├── tests/
│   └── test_retriever.py
├── .env                           # Environment variables
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | NVIDIA NIM API key | Required |
| `QDRANT_URL` | Qdrant database URL | `http://localhost:6333` |
| `NEXT_PUBLIC_API_URL` | Backend API URL (frontend) | `http://localhost:8000` |

### Model Configuration

| Component | Model/Value |
|-----------|-------------|
| LLM | `moonshotai/kimi-k2-instruct` via NVIDIA NIM |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Embedding Dimension | 384 |
| Vector DB Collection | `indonesian_legal_docs` |

---

## 🧪 Testing

### Run Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/

# Q&A test
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Apa itu NIB?"}'
```

---

## 📦 Dependencies

### Backend (Python)
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `qdrant-client` - Vector database client
- `sentence-transformers` - Embedding models
- `rank-bm25` - BM25 sparse retrieval
- `httpx` - Async HTTP client
- `python-multipart` - File upload support
- `PyPDF2` - PDF processing

### Frontend (Node.js)
- `next` - React framework
- `react` - UI library
- `tailwindcss` - CSS framework
- `lucide-react` - Icons

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NVIDIA NIM** for providing free LLM inference
- **Qdrant** for the vector database
- **Hugging Face** for embedding models
- **peraturan.go.id** for Indonesian legal document references

---

## 📞 Support

For issues and questions:
- Open a GitHub Issue
- Email: support@example.com

---

**Built with ❤️ for Indonesian Legal Tech**
