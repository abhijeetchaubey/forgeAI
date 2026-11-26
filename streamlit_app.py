"""
ForgeAI Phase 10: Minimal Streamlit Frontend
Connects to the ForgeAI FastAPI backend and provides:
1. Brief input
2. Document upload
3. Start factory button
4. Clarification question interface
5. Factory execution status
6. Groundability report
7. Generated agent configuration viewer
8. Evaluation scorecard
9. Optimization iteration logs
10. Final report
11. Chat interface to test the generated agent
"""

import os
import time
import httpx
import streamlit as st

API_BASE = os.getenv("FORGEAI_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ForgeAI Agent Factory", page_icon="🛠️", layout="wide")
st.title("🛠️ ForgeAI — Autonomous Agent Factory")
st.caption("Upload domain documents, submit a brief, and let ForgeAI build, evaluate, and optimize your grounded AI agent.")

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "clarification_answers" not in st.session_state:
    st.session_state.clarification_answers = {}

# ---------------------------------------------------------------------------
# Sidebar: 1. Brief Input, 2. Document Upload, 3. Start Factory Button
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Agent Brief & Documents")
    brief = st.text_area(
        "Natural-Language Brief",
        placeholder="e.g., Build an HR Policy Assistant using our handbook, leave policy, and benefits docs...",
        height=130,
    )
    uploaded_files = st.file_uploader(
        "Upload Knowledge Base (.pdf, .csv, .txt)",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
    )
    pass_threshold = st.slider("Pass Threshold", 0.50, 0.95, 0.70, 0.05)
    max_iterations = st.slider("Max Optimization Iterations", 1, 3, 3)

    # 4. Clarification Question Interface (if any surfaced from requirements)
    if st.session_state.job_id:
        try:
            req_resp = httpx.get(f"{API_BASE}/api/jobs/{st.session_state.job_id}/requirements", timeout=5)
            if req_resp.status_code == 200:
                req_data = req_resp.json()
                questions = req_data.get("clarification_questions", [])
                if questions:
                    st.subheader("Clarification Questions")
                    for idx, q in enumerate(questions):
                        ans = st.text_input(q, key=f"clarify_{idx}")
                        if ans:
                            st.session_state.clarification_answers[q] = ans
        except Exception:
            pass

    if st.button("🚀 Start Factory", type="primary", use_container_width=True):
        if not brief.strip():
            st.error("Please enter a natural-language brief.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            try:
                files_payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploaded_files]
                up_resp = httpx.post(f"{API_BASE}/api/documents/upload", files=files_payload, timeout=30)
                up_resp.raise_for_status()
                job_id = up_resp.json()["job_id"]
                st.session_state.job_id = job_id
                st.session_state.chat_history = []

                create_resp = httpx.post(
                    f"{API_BASE}/api/agents/create",
                    json={
                        "job_id": job_id,
                        "brief": brief.strip(),
                        "clarification_answers": st.session_state.clarification_answers or None,
                        "pass_threshold": pass_threshold,
                        "max_iterations": max_iterations,
                    },
                    timeout=15,
                )
                create_resp.raise_for_status()
                st.success(f"Factory started! Job ID: `{job_id}`")
            except Exception as exc:
                st.error(f"Failed to start factory: {exc}")

# ---------------------------------------------------------------------------
# Main Area: Status & Tabs
# ---------------------------------------------------------------------------
job_id = st.session_state.job_id
if not job_id:
    st.info("👈 Enter a brief and upload documents in the sidebar to launch the ForgeAI Factory.")
    st.stop()

# 5. Factory Execution Status
col_s1, col_s2 = st.columns([4, 1])
with col_s2:
    if st.button("🔄 Refresh Status"):
        st.rerun()

try:
    status_resp = httpx.get(f"{API_BASE}/api/jobs/{job_id}/status", timeout=5)
    status_data = status_resp.json() if status_resp.status_code == 200 else {}
except Exception as exc:
    st.error(f"Cannot reach FastAPI backend at {API_BASE}: {exc}")
    st.stop()

with col_s1:
    st.subheader(f"Job `{job_id}` — Status: `{status_data.get('status', 'unknown').upper()}`")
    st.progress(int(status_data.get("progress", 0)) / 100.0, text=status_data.get("current_step", ""))
    if status_data.get("error"):
        st.error(f"Error: {status_data['error']}")

if status_data.get("status") in ("queued", "running"):
    time.sleep(2)
    st.rerun()

tab_ground, tab_agent, tab_score, tab_iter, tab_report, tab_chat = st.tabs([
    "🛡️ Groundability",
    "⚙️ Agent Config",
    "📊 Scorecard",
    "🔁 Iteration Logs",
    "📄 Final Report",
    "💬 Test Chat",
])

# 6. Groundability Report
with tab_ground:
    r = httpx.get(f"{API_BASE}/api/jobs/{job_id}/groundability", timeout=5)
    if r.status_code == 200:
        g = r.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Guaranteed", len(g.get("guaranteed_capabilities", [])))
        c2.metric("Partially Supported", len(g.get("partially_supported_topics", [])))
        c3.metric("Blocked / Unsupported", len(g.get("blocked_unsupported_topics", [])))
        st.json(g)
    else:
        st.info("Groundability report not available yet.")

# 7. Generated Agent Configuration Viewer
with tab_agent:
    r = httpx.get(f"{API_BASE}/api/jobs/{job_id}/agent", timeout=5)
    if r.status_code == 200:
        ag = r.json()
        st.markdown(f"**Agent:** {ag.get('agent_name')} (`v{ag.get('version')}`)")
        st.markdown(f"**Tools Attached:** {', '.join(t['name'] for t in ag.get('selected_tools', [])) or 'None'}")
        st.json(ag)
    else:
        st.info("Agent definition not available yet.")

# 8. Evaluation Scorecard
with tab_score:
    r = httpx.get(f"{API_BASE}/api/jobs/{job_id}/evaluation", timeout=5)
    if r.status_code == 200:
        ev = r.json()
        sc = ev.get("scorecard", {})
        ov = sc.get("overall_evaluation", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Composite Score", f"{ov.get('overall_composite_score', 0):.4f}")
        c2.metric("Pass Rate", f"{ov.get('overall_pass_rate', 0) * 100:.1f}%")
        c3.metric("Passed / Total", f"{ov.get('total_passed', 0)} / {ov.get('total_test_cases', 0)}")
        st.subheader("Category Scores")
        st.json(sc.get("category_scores", {}))
    else:
        st.info("Evaluation scorecard not available yet.")

# 9. Optimization Iteration Logs
with tab_iter:
    r = httpx.get(f"{API_BASE}/api/jobs/{job_id}/evaluation", timeout=5)
    if r.status_code == 200 and r.json().get("iteration_logs"):
        st.json(r.json()["iteration_logs"])
    else:
        st.info("No optimization iteration logs available.")

# 10. Final Report
with tab_report:
    r = httpx.get(f"{API_BASE}/api/jobs/{job_id}/report", timeout=5)
    if r.status_code == 200:
        st.markdown(r.json().get("report_markdown", ""))
    else:
        st.info("Final report not available yet.")

# 11. Chat Interface to Test Generated Agent
with tab_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask the generated agent a question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                chat_resp = httpx.post(
                    f"{API_BASE}/api/jobs/{job_id}/chat",
                    json={"question": prompt},
                    timeout=30,
                )
                chat_resp.raise_for_status()
                data = chat_resp.json()
                ans = data.get("answer", "")
                if data.get("is_refusal"):
                    ans = f"🚫 **Refusal:** {ans}"
                st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
            except Exception as exc:
                st.error(f"Chat error: {exc}")
