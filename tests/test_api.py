"""
Integration Tests for ForgeAI Phase 10: FastAPI Backend
Tests document upload validation, agent creation job queuing, artifact retrieval,
and chat endpoint using FastAPI TestClient.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from api import app, _JOB_STATUS, _RUNTIME_CACHE, JOBS_ROOT
from chat_runner import AgentResponse


client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_documents_validation_and_success(tmp_path):
    # Unsupported file extension -> 400
    resp_bad = client.post(
        "/api/documents/upload",
        files=[("files", ("script.exe", b"binary", "application/octet-stream"))],
    )
    assert resp_bad.status_code == 400

    # Empty file -> 400
    resp_empty = client.post(
        "/api/documents/upload",
        files=[("files", ("empty.txt", b"", "text/plain"))],
    )
    assert resp_empty.status_code == 400

    # Valid upload -> 200
    resp_ok = client.post(
        "/api/documents/upload",
        files=[("files", ("policy.txt", b"Annual PTO is 22 days.", "text/plain"))],
        data={"job_id": "test_job_upload"},
    )
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert data["job_id"] == "test_job_upload"
    assert "policy.txt" in data["uploaded_files"]


def test_create_agent_and_retrieve_artifacts():
    job_id = "test_job_artifacts"
    jdir = JOBS_ROOT / job_id
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "uploads").mkdir(exist_ok=True)
    (jdir / "uploads" / "doc.txt").write_text("Sample text", encoding="utf-8")

    # Populate mock artifacts
    (jdir / "requirements.json").write_text(json.dumps({"project_name": "HR Bot"}), encoding="utf-8")
    (jdir / "groundability_report.json").write_text(json.dumps({"guaranteed_capabilities": ["PTO"]}), encoding="utf-8")
    (jdir / "agent.json").write_text(json.dumps({"agent_name": "HR Bot", "version": "1.0.0"}), encoding="utf-8")
    (jdir / "scorecard.json").write_text(json.dumps({"overall_evaluation": {"overall_composite_score": 0.85}}), encoding="utf-8")
    (jdir / "evaluation_results.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")
    (jdir / "final_quality_report.md").write_text("# Report\nAll passed.", encoding="utf-8")

    with patch("api._run_factory_pipeline"):
        resp = client.post(
            "/api/agents/create",
            json={"job_id": job_id, "brief": "Build an HR assistant."},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    assert client.get(f"/api/jobs/{job_id}/status").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/requirements").json()["project_name"] == "HR Bot"
    assert client.get(f"/api/jobs/{job_id}/groundability").json()["guaranteed_capabilities"] == ["PTO"]
    assert client.get(f"/api/jobs/{job_id}/agent").json()["agent_name"] == "HR Bot"
    assert client.get(f"/api/jobs/{job_id}/evaluation").json()["scorecard"]["overall_evaluation"]["overall_composite_score"] == 0.85
    assert "All passed." in client.get(f"/api/jobs/{job_id}/report").json()["report_markdown"]

    # Test chat endpoint with mocked runtime
    mock_runtime = MagicMock()
    mock_runtime.ask.return_value = AgentResponse(
        answer="Annual PTO is 22 days.",
        is_refusal=False,
        citations=[],
        agent_name="HR Bot",
        tools_used=[],
    )
    _RUNTIME_CACHE[job_id] = mock_runtime
    chat_resp = client.post(f"/api/jobs/{job_id}/chat", json={"question": "How many PTO days?"})
    assert chat_resp.status_code == 200
    assert "22 days" in chat_resp.json()["answer"]
