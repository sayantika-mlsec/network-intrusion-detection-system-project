# Contributing

This is a solo-built project, maintained as if it were a team repo — issues, branches, and PRs are used for every change. This document exists so anyone reviewing the code (or future-me) understands how the project is structured and how work gets done here.

---

## 1. Local Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/sayantika-mlsec/network-intrusion-detection-system-project.git
cd network-intrusion-detection-system-project
```

**2. Isolate your environment**

```bash
python -m venv venv
```

**3. Activate the environment**
- Mac/Linux: `source venv/bin/activate`
- Windows (Command Prompt): `venv\Scripts\activate.bat`
- Windows (PowerShell): `venv\Scripts\Activate.ps1`

**4. Install dependencies**
```bash
pip install -r requirements-serve.txt
pip install -r requirements.txt
```
`requirements-serve.txt` covers what's needed to run the API (used by Docker too). `requirements.txt` adds testing tools (`pytest`, `httpx`) and is the exact set CI installs — kept separate so CI doesn't carry serving-only concerns it doesn't need, and vice versa.

If you're retraining the model rather than using the committed artifacts, use `requirements-dev.txt` instead — it adds `duckdb`, `optuna`, `mlflow`, and `evidently` on top:
```bash
pip install -r requirements-dev.txt
```

**5. ML artifacts**

`nids_pipeline.pkl` and `nids_label_encoder.pkl` are committed to the repo (not gitignored) — they're needed by CI for the integration test suite to load a real model on every push, rather than retraining from scratch each run. If you want to retrain locally instead of using the committed artifacts:
```bash
python train_model.py
```

**6. Run the local development server**
```bash
uvicorn app:app --reload
```
The API is accessible at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

---

## 2. Testing

The test suite is split into two layers, deliberately:

**Unit tests** (`tests/test_api.py`) — fast, in-memory, no server required:
```bash
pytest tests/test_api.py -v
```
These use FastAPI's `TestClient` and `unittest.mock`, validating Pydantic schemas against a real row from `X_test.csv` while mocking the heavy ML artifacts via `@patch.dict('main.ml_models')`. They're fast but never exercise real deserialization.

**Integration tests** (`tests/test_integration.py`) — real lifespan startup, real model load:
```bash
pytest tests/test_integration.py -v
```
These trigger the actual FastAPI lifespan and verify the full train -> serialize -> load -> predict round-trip. This is intentional: unit tests with mocked artifacts can't catch deserialization bugs that only surface when the real pickled pipeline gets loaded in a fresh environment. That gap was a real bug once — not theoretical.

Both suites run automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).

If you add new endpoints or features, add isolated unit tests following the existing mocking pattern, and extend the integration suite if the change touches model loading or serialization.

---

## 3. Workflow

1. **Issue first.** Every change starts as a GitHub issue with checkbox acceptance criteria, before any code is written.
2. **Branch per feature**, with a descriptive prefix:
   - `feature/your-feature-name`
   - `bugfix/issue-description`
   - `docs/readme-updates`
3. **Commit with issue references.** `Refs #N` while in progress, `Closes #N` on the commit that resolves it.
4. **PR with What / Why / Verification:**
   - **What** changed
   - **Why** (the engineering reasoning, not just the diff — e.g. "switched to Isolation Forest to reduce false positives")
   - **Verification** — which tests were run, and why that's sufficient
5. **Self-review before merge.** Even solo, the PR step forces a re-read of the diff before it lands on `main`.

If your change alters API behavior — a new field on the `NetworkPacket` Pydantic model, a changed response shape — update `README.md` in the same PR.

**Exception:** trivial doc-only fixes (typos, broken links, factual corrections that don't change behavior — e.g. fixing a README description to match the actual pipeline) can go straight to `main` without the full issue -> branch -> PR cycle. Anything that changes behavior, adds a feature, or touches code still follows the full workflow above.

---

## 4. Architecture decisions

Non-trivial design choices are recorded as ADRs in [`docs/adr/`](./docs/adr/), numbered in chronological order — for example, the decision to containerize the serving API with Docker. Check there before assuming something was an oversight. More will be added as future decisions (model choice, imbalance handling, etc.) warrant one.

---

## 5. Reporting Issues

Open an issue using this template:

```
**Describe the Bug or Feature:**
[A clear and concise description]

**Environment Details:**
- OS: [e.g., Ubuntu 22.04, Windows 11]
- Python Version: [e.g., 3.10]
- FastAPI/Uvicorn Version: [e.g., 0.100.0]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Include sanitized JSON payload if applicable]

**Expected vs. Actual Behavior:**
[What you expected vs what actually happened]

**Error Logs:**
[Paste logs here. Scrub any sensitive network data, hashes, or internal IP addresses before posting.]
```

## 6. Known limitations

Honest, current limitations are tracked as open GitHub issues rather than buried in code comments. Check the [Issues tab](../../issues) before assuming something is unhandled — it may already be a known, scoped gap with reasoning attached.

If you spot something not already listed — a bug, an edge case, a design tradeoff worth questioning — opening an issue is genuinely useful, even on a solo-maintained project. A second set of eyes catching something is exactly how this list grows.