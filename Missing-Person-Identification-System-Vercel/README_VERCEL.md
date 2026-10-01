# Missing Person Identification System — Vercel migration

This folder is a Vercel/FastAPI migration of the original Streamlit project.

## What changed

- `Home.py` Streamlit UI was replaced by `frontend/index.html`.
- FastAPI is the Python API in `api/index.py`.
- Face-landmark extraction still uses MediaPipe.
- Matching still uses scikit-learn KNN with the original distance-threshold approach.
- Local SQLite is retained only for local development.
- Production should use a managed PostgreSQL database via `DATABASE_URL`.
- Case/sighting images are stored as database bytes in this migration so the demo does not depend on Vercel's ephemeral filesystem.
- Video upload and the original Folium/Streamlit chatbot are not included in this first Vercel migration.

## Security

Do NOT commit the original `login_config.yml` to the Vercel repository. Put authentication users in the `AUTH_USERS_JSON` Vercel environment variable and use a new `SESSION_SECRET`.

The uploaded original project contains a login configuration with bcrypt password hashes and a cookie signing key. Treat the old cookie key as compromised if that file was ever exposed outside your private environment; generate a new session secret for Vercel.

## Local run

Python 3.12 is used because the current MediaPipe package publishes Linux wheels/classifiers through Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set SESSION_SECRET=dev-secret
set AUTH_USERS_JSON={"...":"..."}
uvicorn api.index:app --reload
```

Open http://127.0.0.1:8000

## Vercel

1. Push this folder into a new branch/repository or replace the application files in your repository.
2. In Vercel, import the GitHub repository.
3. Keep the framework as Other/auto-detect; do not select the Streamlit or FastAPI preset for `Home.py`.
4. Add these Environment Variables:
   - `SESSION_SECRET`
   - `DATABASE_URL`
   - `AUTH_USERS_JSON`
5. Deploy.
6. Test `/api/health`, then open `/`.

## Database

The production database must be reachable over the network. The old `sqlite_database.db` should not be relied upon for production persistence.

## Notes

MediaPipe downloads the face-landmarker model into `/tmp` on a cold start. Vercel's Python runtime supports large Functions, but the deployment may still need a larger-function configuration if the dependency bundle exceeds the standard limit.
