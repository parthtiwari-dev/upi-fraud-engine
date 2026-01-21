## Phase 10: Containerization, Render Deployment \& UI

**Goal:** Package the API (and optionally the Streamlit control panel) into Docker images and deploy them to Render, so you have a live, externally accessible fraud decision service.

### Why this is last (on purpose)

- Until Phases 1–9 are solid, you’re just deploying a **broken** system more expensively.
- Debugging feature leakage, alert-budget bugs, and label-delay issues is 10x harder once you’re fighting containers + cloud logs.
- In interviews, talking through “I designed batch/stream consistency + alert-budget backtesting” is worth more than “I clicked Deploy on Render”.

So: **first correctness, then containerization.**

***

### Phase 10: Checklist

**Repo assumptions:** Your repo already has the structure from the roadmap, especially `/api` and (optionally) `/ui`.

#### A. Dockerize the FastAPI service

[ ] Create `Dockerfile.api` at repo root:

- Responsibility: Build an image that:
    - Installs dependencies from `requirements.txt`.
    - Exposes the FastAPI app via Uvicorn on `0.0.0.0:$PORT` (Render sets `PORT`). [^1][^2]

[ ] Add a minimal `.dockerignore` to avoid sending data/model artifacts to the build context (large images = slow deploys):

- Ignore: `data/`, `.venv/`, `__pycache__/`, `*.parquet`, `*.duckdb`, `*.log`, `.git`.

[ ] Decide how models/configs are loaded:

- For a student project, simplest:
    - Bake trained model artifacts into the image (copy `models/*.joblib` in Dockerfile).
    - Load `configs/project.yaml` from the image as well.
- For a more “prod-like” setup:
    - Mount models as a volume or download from remote storage (MLflow artifacts) at startup.

[ ] Locally test the container:

- Build: `docker build -f Dockerfile.api -t upi-fraud-api .` [^2]
- Run: `docker run -p 8000:8000 upi-fraud-api`
- Hit `http://localhost:8000/health` and `/score` using a sample transaction from your offline store.

**Trap (API container):**
If your API image expects `data/offline_store.duckdb` or other large files *inside* the image, you’ll:

- Blow up image size.
- Ship stale data every deploy.
Use the API container **only** for models + code, not for raw data.

***

#### B. Dockerize the Streamlit UI (optional, but good portfolio)

If you build a small control panel in `ui/streamlit_app.py` that:

- Shows recent decisions.
- Plots backtest metrics over time.
- Lets you tweak threshold/budget for “what-if” analysis.

Then:

[ ] Create `Dockerfile.ui`:

- Base `python:3.x-slim`.
- Install `requirements.txt` (reuse or split into `requirements-api.txt` and `requirements-ui.txt` if you want).
- Set `CMD` to `streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0`. [^3][^4]

[ ] Test locally:

- `docker build -f Dockerfile.ui -t upi-fraud-ui .`
- `docker run -p 8501:8501 upi-fraud-ui`
- Visit `http://localhost:8501`.

**Trap (UI container):**
If the UI directly queries your local DuckDB file path, it will work locally but break in Render when the file doesn’t exist. Make the UI call your FastAPI API instead, or read from a small, deployment-friendly store (e.g., a Postgres on Render).

***

#### C. Deploy API to Render

You have two realistic paths:

1. **Render without Docker** (simpler):
    - Connect GitHub repo to Render, use Python runtime with:
        - Build command: `pip install -r requirements.txt` [^1][^5]
        - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT` [^1][^6]
    - No Dockerfile needed in this mode.
2. **Render with Docker** (more explicit infra skills):
    - Push repo with `Dockerfile.api` to GitHub.
    - In Render, create **Web Service → “Deploy an existing image or Dockerfile”**. [^5][^7]
    - Render will:
        - Run `docker build` using your Dockerfile.
        - Start container with the `CMD` defined.

For your portfolio, **path 1 is enough**, unless you explicitly want to showcase Docker.

[ ] In Render dashboard:

- Create a **Web Service** for the FastAPI API.
- Configure:
    - Environment: Python or Docker.
    - Start command (if not Docker).
    - Environment variables:
        - `ENV=production`
        - Paths or secrets if you’re pulling models from elsewhere.

[ ] Confirm deployment:

- Use Render’s URL (e.g., `https://upi-fraud-api.onrender.com/health`).
- Send a `/score` request using `curl` or Postman.

**Trap (Render free tier):**
Cold starts + limited CPU/RAM. [^8][^9]

- Expect the **first request after sleep** to be slow.
- Don’t overcomplicate with heavy backtests inside the API; keep it as a scoring service.

***

#### D. Deploy Streamlit UI to Render (optional)

Two options:

1. Separate Render service:
    - New Web Service, pointing to `Dockerfile.ui` or Python runtime.
    - Start command: `streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`.
2. Same repo, different service:
    - API and UI each get their own web service, sharing the same GitHub repo but different build and start commands.

Then in Streamlit:

- Use the API’s Render URL for all scoring / metrics.
- **Never** hardcode `localhost` for API URL in the deployed version.

***

#### E. Update README and DESIGN

[ ] Add a **Deployment** section in `README.md`:

- “Local run” vs “Render deploy”.
- `docker build` + `docker run` commands.
- Render setup steps in 4–6 bullets.

[ ] In `DESIGN.md`, add a short “Runtime Architecture” section:

- One box for **FastAPI API (Render)**.
- One for **(Optional) Streamlit UI (Render)**.
- One for **Local/Cloud DB + feature store**.
- Arrows showing API calls only; no UI talking directly to local files.

***

### Where this fits in your learning story

- Phases 0–9: Prove you understand **fraud systems, time correctness, batch vs stream, alert budget**.
- Phase 10: Show you can **package \& expose** that system as a real, reachable service.

So the roadmap wasn’t “missing” Docker/Render/Streamlit; it was deliberately focused on **core system design first**. This deployment phase is just the final wrapper.
