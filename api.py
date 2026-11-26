"""
ForgeAI Phase 10: FastAPI Backend
Exposes clean REST endpoints for the ForgeAI Agent Factory:
- Uploading documents
- Creating an agent (running the full factory pipeline in background)
- Checking factory execution status
- Retrieving requirements, groundability, agent definitions, evaluation results, and final reports
- Chatting with generated agents via AgentRuntime
"""

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import List, Dict, Any, Optional
import uuid

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.document_loaders.loaders import load_directory
from src.text_splitter.splitter import split_documents
from src.embeddings.embeddings import get_embedding_model
from src.vectorstore.chroma_store import get_vector_store, add_documents
from intake import conduct_intake
from groundability import analyze_groundability
from generator import generate_agent
from eval_generator import generate_evaluation_set
from optimizer import CriticOptimizerWorkflow
from chat_runner import AgentRuntime
from src.utils.logger import logger

JOBS_ROOT = Path("outputs/jobs")
JOBS_ROOT.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt"}

app = FastAPI(
    title="ForgeAI Agent Factory API",
    description="Domain-independent AI Agent Factory backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry & runtime cache
_JOB_STATUS: Dict[str, Dict[str, Any]] = {}
_RUNTIME_CACHE: Dict[str, AgentRuntime] = {}


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class CreateAgentRequest(BaseModel):
    job_id: str = Field(..., min_length=1, description="Job ID from document upload")
    brief: str = Field(..., min_length=5, description="Natural-language agent brief")
    clarification_answers: Optional[Dict[str, str]] = Field(default=None)
    pass_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_iterations: int = Field(default=3, ge=1, le=3)
    mock_mode: bool = Field(default=False)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question for the generated agent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_dir(job_id: str) -> Path:
    safe_id = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
    if not safe_id:
        raise HTTPException(status_code=400, detail="Invalid job_id.")
    return JOBS_ROOT / safe_id


def _read_json_or_404(path: Path, label: str) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def _update_job(job_id: str, **kwargs) -> None:
    if job_id not in _JOB_STATUS:
        _JOB_STATUS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "current_step": "Initialized",
            "progress": 0,
            "error": None,
            "updated_at": datetime.utcnow().isoformat(),
        }
    _JOB_STATUS[job_id].update(kwargs)
    _JOB_STATUS[job_id]["updated_at"] = datetime.utcnow().isoformat()


def _run_factory_pipeline(req: CreateAgentRequest) -> None:
    """Executes the ForgeAI factory pipeline reusing existing backend modules."""
    job_id = req.job_id
    jdir = _job_dir(job_id)
    upload_dir = jdir / "uploads"
    chroma_dir = jdir / "chroma_db"
    collection_name = f"forgeai_job_{job_id}"

    try:
        _update_job(job_id, status="running", current_step="Loading & indexing documents", progress=15)
        raw_docs = load_directory(upload_dir)
        if not raw_docs:
            raise ValueError("No supported documents found in upload directory.")

        doc_metadata = []
        seen = set()
        for d in raw_docs:
            fname = d.metadata.get("filename", "unknown")
            if fname not in seen:
                seen.add(fname)
                doc_metadata.append({
                    "filename": fname,
                    "file_type": d.metadata.get("file_type", "unknown"),
                    "source": d.metadata.get("source", ""),
                })

        chunks = split_documents(raw_docs)
        embeddings = get_embedding_model()
        vector_store = get_vector_store(
            persist_directory=chroma_dir,
            collection_name=collection_name,
            embedding_model=embeddings,
        )
        add_documents(chunks, vector_store=vector_store)

        _update_job(job_id, current_step="Running Intake Interviewer", progress=30)
        requirements = conduct_intake(
            brief=req.brief,
            documents=raw_docs,
            document_metadata=doc_metadata,
            clarification_answers=req.clarification_answers,
            output_path=jdir / "requirements.json",
        )

        _update_job(job_id, current_step="Running Groundability Guard", progress=45)
        groundability = analyze_groundability(
            requirements=requirements,
            vector_store=vector_store,
            output_path=jdir / "groundability_report.json",
        )

        _update_job(job_id, current_step="Generating Agent Definition", progress=60)
        agent_def = generate_agent(
            requirements=requirements,
            groundability=groundability,
            document_metadata=doc_metadata,
            output_path=jdir / "agent.json",
            collection_name=collection_name,
            persist_directory=str(chroma_dir),
        )

        _update_job(job_id, current_step="Generating Evaluation Set", progress=75)
        eval_set = generate_evaluation_set(
            documents=raw_docs,
            document_dir=upload_dir,
            vector_store=vector_store,
            output_path=jdir / "eval_set.json",
            min_cases=20,
        )

        _update_job(job_id, current_step="Running Judge & Critic-Optimizer Loop", progress=90)
        wf = CriticOptimizerWorkflow(
            original_agent_path=jdir / "agent.json",
            eval_set_path=jdir / "eval_set.json",
            pass_threshold=req.pass_threshold,
            max_iterations=req.max_iterations,
            mock_mode=req.mock_mode,
            iteration_logs_path=jdir / "iteration_logs.json",
            output_dir=jdir,
        )
        opt_report = wf.run()

        final_iter = opt_report["iterations_executed"]
        if final_iter > 0:
            sc_src = jdir / f"scorecard_iter{final_iter}.json"
            ev_src = jdir / f"evaluation_results_iter{final_iter}.json"
            if sc_src.exists():
                shutil.copyfile(sc_src, jdir / "scorecard.json")
            if ev_src.exists():
                shutil.copyfile(ev_src, jdir / "evaluation_results.json")

        # Write final_quality_report.md
        report_lines = [
            f"# Final Quality Report: {agent_def.agent_name}",
            f"- **Job ID:** `{job_id}`",
            f"- **Status:** `{opt_report['status']}`",
            f"- **Final Agent:** `{Path(opt_report['final_agent_path']).name}`",
            f"- **Final Score:** **{opt_report['final_score']:.4f}** (Threshold: {req.pass_threshold:.2f})",
            f"- **Final Pass Rate:** **{opt_report['final_pass_rate'] * 100:.1f}%**",
            f"- **Iterations Executed:** {final_iter} / {req.max_iterations}",
            f"- **Remaining Failures:** {', '.join(opt_report['remaining_failures']) or 'None'}",
        ]
        (jdir / "final_quality_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        _RUNTIME_CACHE.pop(job_id, None)
        _update_job(
            job_id,
            status="completed",
            current_step="Completed",
            progress=100,
            final_agent_path=opt_report["final_agent_path"],
        )
        logger.info("Factory job %s completed successfully.", job_id)

    except Exception as exc:
        logger.error("Factory job %s failed: %s", job_id, str(exc), exc_info=True)
        _update_job(job_id, status="failed", current_step="Failed", error=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ForgeAI Agent Factory API"}


@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    job_id: Optional[str] = Form(default=None),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

    jid = job_id or f"job_{uuid.uuid4().hex[:8]}"
    upload_dir = _job_dir(jid) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}' for '{f.filename}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )
        content = await f.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{f.filename}' is empty.")
        dest = upload_dir / Path(f.filename).name
        dest.write_bytes(content)
        saved_files.append(dest.name)

    _update_job(jid, status="uploaded", current_step="Documents uploaded", progress=10)
    return {"job_id": jid, "uploaded_files": saved_files}


@app.post("/api/agents/create")
def create_agent(req: CreateAgentRequest, background_tasks: BackgroundTasks):
    jdir = _job_dir(req.job_id)
    upload_dir = jdir / "uploads"
    if not upload_dir.exists() or not any(upload_dir.iterdir()):
        raise HTTPException(status_code=400, detail=f"No uploaded documents found for job_id '{req.job_id}'.")

    _update_job(req.job_id, status="queued", current_step="Queued", progress=5, error=None)
    background_tasks.add_task(_run_factory_pipeline, req)
    return {"job_id": req.job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str):
    if job_id not in _JOB_STATUS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _JOB_STATUS[job_id]


@app.get("/api/jobs/{job_id}/requirements")
def get_requirements(job_id: str):
    return _read_json_or_404(_job_dir(job_id) / "requirements.json", "Requirements")


@app.get("/api/jobs/{job_id}/groundability")
def get_groundability(job_id: str):
    return _read_json_or_404(_job_dir(job_id) / "groundability_report.json", "Groundability report")


@app.get("/api/jobs/{job_id}/agent")
def get_agent_definition(job_id: str):
    jdir = _job_dir(job_id)
    # Return latest versioned agent if available, else agent.json
    logs_path = jdir / "iteration_logs.json"
    if logs_path.exists():
        logs = json.loads(logs_path.read_text(encoding="utf-8"))
        final_p = Path(logs.get("final_agent_path", ""))
        if final_p.exists():
            return json.loads(final_p.read_text(encoding="utf-8"))
    return _read_json_or_404(jdir / "agent.json", "Agent definition")


@app.get("/api/jobs/{job_id}/evaluation")
def get_evaluation_results(job_id: str):
    jdir = _job_dir(job_id)
    scorecard = _read_json_or_404(jdir / "scorecard.json", "Scorecard")
    results = _read_json_or_404(jdir / "evaluation_results.json", "Evaluation results")
    iter_logs = (
        json.loads((jdir / "iteration_logs.json").read_text(encoding="utf-8"))
        if (jdir / "iteration_logs.json").exists()
        else None
    )
    return {
        "scorecard": scorecard,
        "evaluation_results": results,
        "iteration_logs": iter_logs,
    }


@app.get("/api/jobs/{job_id}/report")
def get_final_report(job_id: str):
    report_path = _job_dir(job_id) / "final_quality_report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Final report not found.")
    return {"job_id": job_id, "report_markdown": report_path.read_text(encoding="utf-8")}


@app.post("/api/jobs/{job_id}/chat")
def chat_with_agent(job_id: str, req: ChatRequest):
    jdir = _job_dir(job_id)
    if job_id not in _RUNTIME_CACHE:
        agent_path = jdir / "agent.json"
        logs_path = jdir / "iteration_logs.json"
        if logs_path.exists():
            logs = json.loads(logs_path.read_text(encoding="utf-8"))
            cand = Path(logs.get("final_agent_path", ""))
            if cand.exists():
                agent_path = cand
        if not agent_path.exists():
            raise HTTPException(status_code=404, detail="Generated agent not found for this job.")
        _RUNTIME_CACHE[job_id] = AgentRuntime.from_agent_file(agent_path)

    resp = _RUNTIME_CACHE[job_id].ask(req.question)
    return resp.model_dump()
