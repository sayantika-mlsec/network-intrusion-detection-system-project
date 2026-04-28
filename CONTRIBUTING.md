```markdown
# Contributing to the NIDS API

First, thank you for taking the time to contribute! We rely on the community to help harden this Network Intrusion Detection System (NIDS) API. 

Whether it is a bug fix, a new SHAP visualization, or a performance upgrade, please follow the guidelines below to ensure a smooth, asynchronous collaboration process.

---

## 1. Local Setup & Installation

To run the NIDS API locally, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/sayantika-mlsec/network-intrusion-detection-system-project.git](https://github.com/sayantika-mlsec/network-intrusion-detection-system-project.git)
cd network-intrusion-detection-system-project
```

**2. Isolate your environment**
We strongly recommend using a virtual environment to avoid dependency conflicts.
```bash
python -m venv venv
```

**3. Activate the environment**
* **Mac/Linux:** `source venv/bin/activate`
* **Windows (Command Prompt):** `venv\Scripts\activate.bat`
* **Windows (PowerShell):** `venv\Scripts\Activate.ps1`

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Generate ML Artifacts**
To keep this repository lightweight, compiled model binaries are ignored via `.gitignore`. You must train the model locally to generate the artifacts.
```bash
python train_model.py
```
*(Verify that `nids_pipeline.pkl` and `nids_label_encoder.pkl` are generated in your root directory before proceeding).*

**6. Run the local development server**
```bash
uvicorn main:app --reload
```
The API will be accessible at `http://127.0.0.1:8000`. 
View the interactive API documentation at `http://127.0.0.1:8000/docs`.

---

## 2. Reporting Issues

If you find a bug or have a feature request, please open an issue. To help us resolve it quickly, please copy and paste the following template into your issue description and fill it out:

**Issue Template:**
```text
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
[Paste logs here. WARNING: Scrub all sensitive network data, proprietary hashes, or internal IP addresses before posting!]
```

---

## 3. Pull Request (PR) Process

We welcome contributions of all sizes. To get your code merged, follow this workflow:

**1. Fork and Branch**
Always fork the repository and create a new branch for your work. Use a descriptive prefix:
* `feature/your-feature-name`
* `bugfix/issue-description`
* `docs/readme-updates`

**2. Pass the Fire Drills (Testing)**
We use FastAPI's `TestClient` and Python's `unittest.mock` to perform rapid, in-memory testing. You do not need to have the Uvicorn server running to execute the test suite. 

Ensure you have not broken the existing ML infrastructure or schema validation by running our test script:
```bash
pytest test_api.py -v
```
*Note for Contributors:* Our tests use a real dataset row (`X_test.csv`) to validate Pydantic schemas, while utilizing `@patch.dict('main.ml_models')` to mock the heavy ML artifacts. If you add new endpoints or ML features, please include relevant isolated tests following this architecture.

**3. Update the Documentation**
If your PR changes the API's behavior—such as adding a new feature to the Pydantic `NetworkPacket` model or altering the JSON response—you must update the `README.md`.

**4. Write a Clear PR Description**
When opening your PR, provide context:
* Link to the issue your PR solves using GitHub automation keywords (e.g., `Closes #12`).
* Briefly explain your technical approach (e.g., "Switched to Isolation Forest to reduce false positives").
* Include a screenshot if your PR alters any visual outputs.
```
