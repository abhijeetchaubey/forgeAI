# ForgeAI

ForgeAI is a document-grounded AI agent factory. Upload a brief and PDF, CSV, or
TXT documents; ForgeAI builds an assistant, checks its capabilities against the
uploaded evidence, generates an evaluation set, and runs an optimization
workflow.

## Contents

- [Features](#features)
- [Factory workflow](#factory-workflow)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Run the web application](#run-the-web-application)
- [Run the command-line demos](#run-the-command-line-demos)
- [Configuration](#configuration)
- [API endpoints](#api-endpoints)
- [Tests](#tests)
- [Generated files](#generated-files)
- [Troubleshooting](#troubleshooting)

## Features

- Load PDF, CSV, and TXT documents with source metadata.
- Split documents into chunks and create local Hugging Face embeddings.
- Store and retrieve document chunks with ChromaDB.
- Extract requirements from a natural-language brief.
- Audit requested capabilities against retrieved document evidence.
- Generate a declarative agent definition with selected tools and refusal rules.
- Generate evaluation cases and run judge and critic-optimizer workflows.
- Test generated agents in the Streamlit chat interface or Python runtime.

## Factory workflow

1. **Upload:** Provide a brief and one or more domain documents.
2. **Intake:** Extract the audience, scope, expectations, and refusal topics.
3. **Groundability:** Check requested capabilities against document evidence.
4. **Agent generation:** Create and validate an agent definition and tool set.
5. **Evaluation:** Generate test cases from the available document corpus.
6. **Optimization:** Score the agent and iterate on its configuration.
7. **Chat:** Ask the completed agent questions and review its evidence-based answers.

The evaluation generator adapts to the corpus. For example, it does not request
cross-document test cases when only one source document was uploaded.

## Project structure

```text
forgeAI/
├── api.py                     # FastAPI backend and job endpoints
├── streamlit_app.py           # Web interface
├── config.py                  # Environment-based settings
├── intake.py                  # Requirements extraction
├── groundability.py           # Evidence-based capability audit
├── generator.py               # Agent definition generation
├── eval_generator.py          # Evaluation-set generation
├── optimizer.py               # Judge and critic-optimizer workflow
├── chat_runner.py             # Generic agent runtime
├── run_pipeline.py            # Document ingestion and search demo
├── run_demos.py               # HR and ship-spare factory demos
├── src/
│   ├── document_loaders/       # PDF, CSV, and TXT loaders
│   ├── embeddings/             # Local Hugging Face embeddings
│   ├── llm/                    # Groq client
│   ├── text_splitter/          # Document chunking
│   ├── tools/                  # Runtime tools
│   └── vectorstore/            # ChromaDB integration
├── data/
│   ├── sample_docs/            # Small ingestion examples
│   └── demos/                  # HR and procurement demo documents
├── outputs/jobs/               # Per-job uploads and generated artifacts
├── prompts/                    # LLM prompt templates
├── tests/                      # Unit and integration tests
├── requirements.txt
└── .env.example                # Configuration template
```

## Setup

Use PowerShell from the repository root.

### Create the environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `.env` does not already exist, create it from the template:

```powershell
Copy-Item .env.example .env
```

Set a valid `GROQ_API_KEY` in `.env`. Keep `.env` private; it is excluded from
version control. The local embedding model may download the first time it is
used.

## Run the web application

Start the backend in one PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the Streamlit frontend in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8502
```

Open [http://localhost:8502](http://localhost:8502).

### Use the factory

1. Enter a brief describing the assistant, its audience, and its boundaries.
2. Upload the documents the assistant should use.
3. Set the pass threshold and maximum optimization iterations.
4. Select **Start Factory** and monitor the job status.
5. Review the Groundability, Agent Config, Scorecard, Iteration Logs, and Final
   Report tabs.
6. Use **Test Chat** to ask the generated agent questions.

For example, the HR demo documents are in `data/demos/hr_policy/`. Upload the
handbook, leave policy, and benefits guide together for an HR assistant that can
answer across all three sources.

### Use a different backend port

The frontend defaults to `http://127.0.0.1:8000`. If that port is occupied, run
the backend on another port, such as `8001`, and set the frontend URL before
starting Streamlit:

```powershell
$env:FORGEAI_API_URL = "http://127.0.0.1:8001"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8502
```

## Run the command-line demos

`run_demos.py` runs an end-to-end HR Policy Assistant demo, a Ship-Spare
Procurement Assistant demo, or both:

```powershell
# Run both demos
.\.venv\Scripts\python.exe run_demos.py

# Run only the HR demo with one optimization iteration
.\.venv\Scripts\python.exe run_demos.py --only demo1 --max-iterations 1

# Run only the procurement demo
.\.venv\Scripts\python.exe run_demos.py --only demo2
```

The script creates its demo datasets and writes results under `outputs/`.
These workflows make LLM requests and use provider quota.

`run_pipeline.py` is a smaller document-ingestion and semantic-search
demonstration:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

## Configuration

Copy `.env.example` to `.env` and set values for your environment.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM provider (`groq` or `gemini`) | `groq` |
| `GROQ_API_KEY` | Groq API key | Empty; required for Groq |
| `GROQ_MODEL` | Groq model identifier | `openai/gpt-oss-20b` |
| `GROQ_TEMPERATURE` | Generation temperature | `0.0` |
| `CHUNK_SIZE` | Maximum document chunk size in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks in characters | `200` |
| `EMBEDDING_MODEL_NAME` | Local embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB persistence directory | `./data/chroma_db` |
| `CHROMA_COLLECTION_NAME` | Default ChromaDB collection | `forgeai_documents` |
| `TOP_K` | Number of chunks retrieved for a query | `4` |
| `LOG_LEVEL` | Application logging level | `INFO` |

The `GEMINI_API_KEY` and `GEMINI_MODEL` settings are used when
`LLM_PROVIDER=gemini`.

## API endpoints

The FastAPI backend exposes these main routes:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Check backend health |
| `POST` | `/api/documents/upload` | Upload documents and create a job |
| `POST` | `/api/agents/create` | Start the factory pipeline |
| `GET` | `/api/jobs/{job_id}/status` | Get job status and errors |
| `GET` | `/api/jobs/{job_id}/requirements` | Get extracted requirements |
| `GET` | `/api/jobs/{job_id}/groundability` | Get capability audit results |
| `GET` | `/api/jobs/{job_id}/agent` | Get the generated agent definition |
| `GET` | `/api/jobs/{job_id}/evaluation` | Get scorecard and evaluation results |
| `GET` | `/api/jobs/{job_id}/report` | Get the final report |
| `POST` | `/api/jobs/{job_id}/chat` | Chat with the generated agent |

## Tests

Run the test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Some integration tests may call the configured LLM provider and consume quota.

## Generated files

Factory jobs write their uploads and results to `outputs/jobs/<job_id>/`,
including requirements, groundability results, the agent definition,
evaluation cases, scorecards, iteration logs, and the final report. The API
also provides routes for retrieving these artifacts.

The command-line demos write to `outputs/demo1_hr_policy/` and
`outputs/demo2_ship_spare/`.

## Troubleshooting

### Groq returns HTTP 429

A 429 indicates a provider-side rate or token quota limit. Wait for the quota
window to reset, use a model with available quota, or adjust the Groq plan.
ForgeAI stops agent-generation retries on a 429 rather than repeatedly
requesting the exhausted model. A failed job must be started again as a new
job after quota is available.

### The frontend cannot reach the backend

Check that Uvicorn is running, that `FORGEAI_API_URL` matches the backend port,
and that `GET /health` returns a healthy response.

### The agent refuses a question

The runtime answers from uploaded evidence and refuses unsupported questions.
Upload the source documents that contain the requested information and start a
new job.
