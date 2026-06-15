ADR: Containerization Strategy for NIDS FastAPI Application
Status: Accepted
Context The Network Intrusion Detection System (NIDS) serving API needs to transition from a host-dependent execution model to a containerized architecture. This transition must provide consistent execution across environments (development, staging, production) while strictly enforcing security boundaries to prevent the leakage of proprietary training data, background scripts, and MLflow tracking artifacts into the final production image. Furthermore, the build process must be optimized for speed and the final artifact minimized for efficient deployment.
Decision We will containerize the FastAPI serving application using Docker, adhering to the following specific architectural constraints:

* Base Image: Utilize `python:3.11-slim` to minimize the container's attack surface and footprint by excluding unnecessary OS-level packages.
* Layer Caching Strategy: Execute `COPY requirements-serve.txt .` and `RUN pip install --no-cache-dir` prior to copying the application source code (`COPY . .`). This structure isolates the heavy dependency installation step, preventing minor application code changes from busting the Docker cache and triggering full dependency rebuilds.
* Security Boundary via `.dockerignore`: Implement a strict `.dockerignore` file acting as a security gate. It will explicitly block `data/`, `mlflow.db`, `mlartifacts/`, `scratch.ipynb`, `train_model.py`, and global `*.pkl` files from entering the build context to ensure proprietary assets are not baked into the image.
* Dynamic Model Loading: Allow the default production model (`!nids_pipeline.pkl` and `!nids_label_encoder.pkl`) through the `.dockerignore` gate. The application will use `os.getenv("MODEL_PATH", "nids_pipeline.pkl")` and `os.getenv("ENCODER_PATH","nids_label_encoder.pkl")` to permit seamless runtime overrides of the model file via Docker environment variables without requiring code modifications.
* Execution Command: Use the Exec form for the startup command (`CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]`) rather than Shell form. This binds the server to accept external traffic and ensures graceful shutdown by allowing POSIX signals (like `SIGTERM`) to pass directly to Uvicorn.
Consequences
Positive:

* Predictability: The "works on my machine" anti-pattern is eliminated; the environment is entirely codified.
* Security: Risk of exposing training datasets or experimental model artifacts in the deployed registry is mitigated.
* Performance: Local iterative development builds will execute in seconds due to optimized layer caching.
* Flexibility: DevOps can seamlessly swap active models in production by injecting new paths via the `-e MODEL_PATH` environment variable at runtime.
Negative:

* Maintenance Overhead: Engineers must remember to update the explicit whitelist in `.dockerignore` if the default fallback model filename changes.
* Host Dependency: MLflow tracking data generated during live container execution will be lost on container termination unless explicitly mapped to a host volume (which is outside the scope of this specific serving-app containerization).