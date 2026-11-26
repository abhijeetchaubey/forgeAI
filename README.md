# ForgeAI - AI Factory Platform

ForgeAI is an industrial AI Factory and Document Intelligence Platform.

- **Phase 1: Project Foundation & Document Intelligence** (PDF/CSV/TXT loaders, all-MiniLM-L6-v2 local embeddings, ChromaDB vector store).
- **Phase 2: Intake Interviewer & Groundability Guard** (Requirements extraction, evidence-based groundability audit, capabilities barrier).
- **Phase 3: Dynamic Agent Generator** (Declarative, domain-independent agent synthesis with Pydantic validation, tool selection, and controlled retries).
- **Phase 4: Generic Agent Runtime** (Generic agent loader, ChromaDB context retrieval, factual grounding, tool attachment, source citations, and refusal enforcement).
- **Phase 5: Automatic Evaluation Generator** (Automated generation of 20+ test cases across answerable, multi-document, ambiguous, and out-of-scope categories, with ground verification and prompt archiving).

---

## Architecture Overview

```
forgeAI/
├── .env                              # Active configuration variables
├── .env.example                      # Configuration template
├── config.py                         # Pydantic configuration loader & strict validator
├── intake.py                         # Phase 2: Intake Interviewer (produces requirements.json)
├── groundability.py                  # Phase 2: Groundability Guard (produces groundability_report.json)
├── generator.py                      # Phase 3: Dynamic Agent Generator (produces agent.json)
├── agent_loader.py                   # Phase 4: Generic Agent Loader & Tool Factory
├── chat_runner.py                    # Phase 4: Domain-Independent Agent Runtime & Interactive Chat
├── eval_generator.py                 # Phase 5: Automatic Evaluation Generator (produces eval_set.json)
├── requirements.json                 # Generated structured requirements specification
├── groundability_report.json         # Generated groundability verification report
├── agent.json                        # Generated declarative agent definition
├── eval_set.json                     # Generated 20+ evaluation test cases
├── prompts/                          # Archived evaluation prompt templates
│   ├── eval_system_prompt.txt        # System prompt template for test case generation
│   └── eval_generation_prompt.txt    # Task prompt templates (grounded & boundary cases)
├── requirements.txt                  # Project dependencies
├── run_pipeline.py                   # Demonstration CLI script for the end-to-end pipeline
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration module interface
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py                 # Centralized structured logging
│   ├── llm/
│   │   ├── __init__.py
│   │   └── groq_client.py            # ChatGroq initialization & validation
│   ├── document_loaders/
│   │   ├── __init__.py
│   │   └── loaders.py                # PDF, TXT, CSV loaders with metadata extraction
│   ├── text_splitter/
│   │   ├── __init__.py
│   │   └── splitter.py               # RecursiveCharacterTextSplitter with chunk metadata
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embeddings.py             # HuggingFace all-MiniLM-L6-v2 local embeddings
│   └── vectorstore/
│       ├── __init__.py
│       └── chroma_store.py           # ChromaDB storage, persistence, & semantic search
├── data/
│   ├── sample_docs/                  # Sample PDF, CSV, and TXT files for verification
│   └── chroma_db/                    # Persistent Chroma vector store (SQLite + indices)
└── tests/
    ├── conftest.py                   # Pytest fixtures and mock data
    ├── test_config.py                # Configuration validation tests
    ├── test_groq.py                  # Groq LLM initialization tests
    ├── test_loaders.py               # Document loader tests
    ├── test_splitter.py              # Text chunking and metadata tests
    ├── test_vectorstore.py           # Chroma persistence & search tests
    ├── test_end_to_end.py            # Phase 1 integration verification
    ├── test_intake.py                # Phase 2: Intake interview & requirements schema tests
    ├── test_groundability.py         # Phase 2: Groundability classification & guard tests
    ├── test_generator.py             # Phase 3: Agent schema validation & tool selection tests
    ├── test_runtime.py               # Phase 4: Multi-agent runtime & citation/refusal tests
    └── test_eval_generator.py        # Phase 5: Automatic evaluation generator tests
```

---

## Phase 5: Automatic Evaluation Generator (`eval_generator.py`)

Generates at least 20 diverse, verified evaluation test cases based strictly on uploaded documents and their metadata:

### Test Case Schema (`eval_set.json`)
Each case in `eval_set.json` contains:
1. `test_id`: Unique identifier (e.g. `EVAL-001`).
2. `question`: The evaluation query.
3. `category`: One of four test categories:
   - **`answerable`**: Single-document factual inquiries directly supported by document chunks.
   - **`multi_document`**: Complex cross-referencing questions requiring synthesis across two or more distinct documents (e.g., comparing PDF cutoff temperature with CSV telemetry).
   - **`ambiguous`**: Inquiries referencing domain concepts but omitting critical entity identifiers or timeframes, testing clarification behavior.
   - **`out_of_scope`**: Inquiries clearly absent from the corpus, verifying strict refusal behavior.
4. `expected_facts`: List of concrete facts required in the response (or refusal reasoning).
5. `expected_source_document`: Document names where evidence is located.
6. `expected_page_number`: Target page or row reference.
7. `should_refuse`: `true` for out-of-scope inquiries, `false` for answerable/multi-document.
8. `evaluation_criteria`: Explicit rubric for grading correctness.

### Prompt Templates Archive (`prompts/`)
- `prompts/eval_system_prompt.txt`: System instructions enforcing factual grounding and category distribution.
- `prompts/eval_generation_prompt.txt`: Task prompts for in-scope grounded cases and boundary cases.

---

## Execution Instructions

### Run Automated Tests (46/46 Passing)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Run Automatic Evaluation Generator
```powershell
.\.venv\Scripts\python.exe eval_generator.py
```

### Launch Interactive Agent Chat Session
```powershell
.\.venv\Scripts\python.exe chat_runner.py agent.json
```

---

## Phase 4: Generic Agent Runtime

### 1. Generic Agent Loader (`agent_loader.py`)
- Loads and parses any valid `AgentDefinition` (`agent.json`).
- Connects to the agent's target ChromaDB collection using its configured embedding model.
- Initializes Groq `ChatGroq`.
- Dynamically creates and attaches **only** the tools specified in `selected_tools`.

### 2. Chat Runner & Runtime (`chat_runner.py`)
- **Domain-Independent**: Zero hardcoded domain chatbots (no `HRChatbot` or `ProcurementChatbot` classes).
- **Factual Grounding**: Retrieves document chunks per question and evaluates similarity against `similarity_threshold`.
- **Refusal Enforcement**: Automatically refuses unsupported topics or queries where no relevant document evidence exists.
- **Source Citations**: Returns structured `Citation` records with document filename, page number, row number, and similarity score.
- **Interactive Mode**: Provides rich terminal chat via `python chat_runner.py agent.json`.

---

## Execution Instructions

### Run Full Automated Test Suite (41/41 Passing)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Launch Interactive Agent Chat Session
```powershell
.\.venv\Scripts\python.exe chat_runner.py agent.json
```

### Run Multi-Agent Runtime Programmatically
```python
from chat_runner import AgentRuntime

# Load any declarative agent definition
runtime = AgentRuntime.from_agent_file("agent.json")

# Grounded inquiry
response = runtime.ask("What is the emergency cutoff temperature?")
print("Answer:", response.answer)
print("Citations:", response.citations)

# Unsupported inquiry (triggers refusal policy)
refusal_response = runtime.ask("What are the employee 401k matching percentages?")
print("Refusal:", refusal_response.answer)
print("Is Refused:", refusal_response.is_refusal)
```

---

## Phase 3: Dynamic Agent Generator (`generator.py`)

Synthesizes a declarative, domain-independent `AgentDefinition` from:
1. Validated requirements (`requirements.json`)
2. Groundability report (`groundability_report.json`)
3. Document metadata
4. Available tool registry

### Generated Agent Definition Elements (`agent.json`)
1. **Agent Name & Description**: Synthesized based on operational scope.
2. **System Prompt**: Rigorous instruction set defining persona, scope, citation formats, and refusal guidelines.
3. **Supported & Unsupported Topics**: Supported topics strictly mirror `guaranteed_capabilities`; unsupported topics combine blocked groundability topics and `must_refuse_topics`.
4. **Audience & Tone**: Target user persona and response style.
5. **Retrieval Configuration**: `embedding_model`, `chunk_size`, `chunk_overlap`, `top_k`, and `similarity_threshold`.
6. **Selected Tools**: Domain-appropriate subset selected from the tool registry (e.g. `document_retriever`, `tabular_data_inspector`, `engineering_calculator`, `incident_ticket_creator`).
7. **Refusal Policy**: Standardized handling for out-of-scope, ungrounded, or prohibited queries.
8. **Output Contract**: Markdown formatting, citation syntax `[source: {filename}, page: {page_number}]`, and factual constraints.

### Validation & Reliability
- Strict Pydantic models: `AgentDefinition`, `RetrievalConfig`, `RefusalPolicy`, `OutputContract`, `ToolSpec`.
- Controlled retry loop (up to 3 attempts) with error feedback to recover from invalid JSON or missing fields.

---

## Execution Instructions

### Run Automated Tests (36/36 Passing)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Run Intake Interviewer
```powershell
.\.venv\Scripts\python.exe intake.py
```

### Run Groundability Guard
```powershell
.\.venv\Scripts\python.exe groundability.py
```

### Run Dynamic Agent Generator
```powershell
.\.venv\Scripts\python.exe generator.py
```

### Run Full Ingestion & Search Pipeline
```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

---

## Phase 2 Components

### 1. Intake Interviewer (`intake.py`)
- Analyzes a natural-language brief and uploaded document metadata/content.
- Uses ChatGroq with Pydantic structured output to extract:
  - `intended_audience`
  - `scope`
  - `tone`
  - `user_expectations`
  - `must_answer_topics`
  - `must_refuse_topics`
  - `clarification_questions` (strictly limited to at most 5 questions when genuine ambiguity exists).
- Serializes and validates the resulting specification into `requirements.json`.

### 2. Groundability Guard (`groundability.py`)
- Retrieves actual document evidence chunks from ChromaDB for each requested topic.
- Evaluates topic coverage strictly against document evidence rather than relying only on LLM assumptions.
- Classifies each topic as:
  - `supported`: Sufficient factual evidence in uploaded documents directly answers the topic.
  - `partially_supported`: Mentions the topic or component, but lacks critical operational procedures or parameters.
  - `unsupported`: No relevant or negligible document evidence found.
- Identifies missing information for ungrounded or partially grounded topics.
- **Strict Capability Enforcement**: Only `supported` topics are included in `guaranteed_capabilities`. Unsupported topics are blocked and quarantined in `blocked_unsupported_topics`.
- Serializes the audit to `groundability_report.json`.

---

## Execution Instructions

### Run Automated Tests (31/31 Passing)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Run Intake Interviewer
```powershell
.\.venv\Scripts\python.exe intake.py
```

### Run Groundability Guard
```powershell
.\.venv\Scripts\python.exe groundability.py
```

### Run Full Phase 1 Ingestion Pipeline
```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

---

## Key Features & Compliance

1. **Independent & Reusable Modules**: Each module (`llm`, `document_loaders`, `text_splitter`, `embeddings`, `vectorstore`) has clear single responsibilities and clean public interfaces.
2. **Metadata Preservation**: Loaders extract and preserve `filename`, `source`, `page_number` (1-indexed), and `file_type`. CSV records include `row_number`. Chunks include `chunk_index`.
3. **Local Embeddings**: Embeddings run locally on CPU using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional). Neither OpenAI nor Ollama are used.
4. **Groq for Generation**: `ChatGroq` is used exclusively for LLM generation with model `llama-3.3-70b-versatile` and temperature configuration.
5. **Persistence**: ChromaDB stores embeddings and metadata to disk (`./data/chroma_db/chroma.sqlite3`).
6. **Error Handling & Logging**: Comprehensive logging via `src.utils.logger` with graceful error handling across all file operations and search queries.

---

## Configuration (`.env`)

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API Key | `""` |
| `GROQ_MODEL` | Groq model identifier | `llama-3.3-70b-versatile` |
| `GROQ_TEMPERATURE` | Temperature for generation (0.0 to 2.0) | `0.0` |
| `CHUNK_SIZE` | Maximum chunk size in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks in characters | `200` |
| `EMBEDDING_MODEL_NAME`| Local HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIRECTORY` | Chroma database persistence directory | `./data/chroma_db` |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection name | `forgeai_documents` |
| `TOP_K` | Number of documents to retrieve in search | `4` |
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

---

## Execution Instructions

### 1. Run the Complete Pipeline Demonstration
Executes document loading (PDF, CSV, TXT), text chunking, embedding generation, ChromaDB persistence, and semantic searches:

```bash
.\.venv\Scripts\python.exe run_pipeline.py
```

### 2. Run the Automated Test Suite
Executes unit and integration tests across all modules:

```bash
.\.venv\Scripts\pytest.exe -v
```
#   f o r g e A I  
 