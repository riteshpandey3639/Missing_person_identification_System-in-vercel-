
import base64
import json
import os
import secrets
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import bcrypt
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Cookie
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sqlmodel import Field, SQLModel, Session, create_engine, select
from sqlalchemy import Column, LargeBinary

APP_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = APP_ROOT / "frontend"
MODEL_PATH = Path("/tmp/face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

app = FastAPI(title="Missing Person Identification System API", version="1.0.0")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{APP_ROOT / 'local.sqlite3'}")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)


class RegisteredCase(SQLModel, table=True):
    id: str = Field(primary_key=True)
    submitted_by: str = Field(index=True)
    name: str
    father_name: Optional[str] = None
    age: Optional[str] = None
    complainant_name: str
    complainant_mobile: Optional[str] = None
    complainant_email: Optional[str] = None
    adhaar_card: Optional[str] = None
    last_seen: str
    address: Optional[str] = None
    city: Optional[str] = Field(default=None, index=True)
    description: Optional[str] = None
    face_mesh: str
    image_data: Optional[bytes] = Field(default=None, sa_column=Column(LargeBinary))
    submitted_on: datetime = Field(default_factory=datetime.utcnow)
    status: str = "NF"
    matched_with: Optional[str] = None


class PublicSubmission(SQLModel, table=True):
    id: str = Field(primary_key=True)
    submitted_by: str
    face_mesh: str
    location: str
    mobile: str
    email: Optional[str] = None
    status: str = "NF"
    birth_marks: Optional[str] = None
    image_data: Optional[bytes] = Field(default=None, sa_column=Column(LargeBinary))
    submitted_on: datetime = Field(default_factory=datetime.utcnow)


@app.on_event("startup")
def startup():
    SQLModel.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
SESSION_SECRET = os.getenv("SESSION_SECRET", "CHANGE_ME_IN_VERCEL")
SESSION_TTL_HOURS = 12


def _users():
    raw = os.getenv("AUTH_USERS_JSON", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("AUTH_USERS_JSON is not valid JSON")


def _make_token(username: str) -> str:
    # Simple signed session token using HMAC-style construction.
    import hmac, hashlib
    payload = {
        "u": username,
        "exp": int((datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)).timestamp()),
        "n": secrets.token_hex(8),
    }
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    sig = hmac.new(SESSION_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def _verify_token(token: Optional[str]):
    if not token or "." not in token:
        return None
    import hmac, hashlib
    body, sig = token.rsplit(".", 1)
    expected = hmac.new(SESSION_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        if int(payload["exp"]) < int(datetime.utcnow().timestamp()):
            return None
        return payload["u"]
    except Exception:
        return None


def current_user(session: Optional[str] = Cookie(default=None, alias="mpis_session")):
    username = _verify_token(session)
    if not username:
        raise HTTPException(status_code=401, detail="Authentication required")
    users = _users()
    if username not in users:
        raise HTTPException(status_code=401, detail="Invalid session")
    return username, users[username]


# ---------------------------------------------------------------------------
# MediaPipe face landmarks
# ---------------------------------------------------------------------------
_detector = None


def _ensure_detector():
    global _detector
    if _detector is not None:
        return _detector
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    _detector = mp_vision.FaceLandmarker.create_from_options(options)
    return _detector


def extract_faces(image_bytes: bytes, max_faces: int = 5):
    from PIL import Image
    import mediapipe as mp
    image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)
    h, w = arr.shape[:2]
    detector = _ensure_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    result = detector.detect(mp_image)
    faces = []
    for lm_list in result.face_landmarks[:max_faces]:
        xs = [lm.x * w for lm in lm_list]
        ys = [lm.y * h for lm in lm_list]
        bbox = [
            max(0, int(min(xs))), max(0, int(min(ys))),
            min(w, int(max(xs))), min(h, int(max(ys)))
        ]
        landmarks = [coord for lm in lm_list for coord in (lm.x, lm.y, lm.z)]
        faces.append({"landmarks": landmarks, "bbox": bbox})
    return faces


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


class MatchRequest(BaseModel):
    threshold: float = 3.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "service": "Missing Person Identification System"}


@app.post("/api/login")
def login(data: LoginRequest):
    users = _users()
    user = users.get(data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Username or password is incorrect")
    stored_hash = user.get("password_hash", "")
    try:
        valid = bcrypt.checkpw(data.password.encode(), stored_hash.encode())
    except Exception:
        valid = False
    if not valid:
        raise HTTPException(status_code=401, detail="Username or password is incorrect")
    from fastapi.responses import JSONResponse
    response = JSONResponse({
        "ok": True,
        "username": data.username,
        "name": user.get("name", data.username),
        "role": user.get("role", "Officer"),
        "city": user.get("city", ""),
        "area": user.get("area", ""),
    })
    response.set_cookie(
        "mpis_session", _make_token(data.username),
        httponly=True, secure=True, samesite="lax",
        max_age=SESSION_TTL_HOURS * 3600
    )
    return response


@app.post("/api/logout")
def logout():
    from fastapi.responses import JSONResponse
    response = JSONResponse({"ok": True})
    response.delete_cookie("mpis_session")
    return response


@app.get("/api/me")
def me(user=Depends(current_user)):
    username, profile = user
    return {
        "username": username,
        "name": profile.get("name", username),
        "role": profile.get("role", "Officer"),
        "city": profile.get("city", ""),
        "area": profile.get("area", ""),
    }


@app.post("/api/cases")
async def create_case(
    name: str = Form(...),
    father_name: str = Form(""),
    age: str = Form(""),
    mobile: str = Form(""),
    aadhaar: str = Form(""),
    address: str = Form(""),
    city: str = Form(""),
    birth_marks: str = Form(""),
    last_seen: str = Form(...),
    description: str = Form(""),
    complainant_name: str = Form(...),
    complainant_phone: str = Form(...),
    complainant_email: str = Form(""),
    face_index: int = Form(0),
    image: UploadFile = File(...),
    user=Depends(current_user),
):
    username, profile = user
    if len(complainant_phone.strip()) != 10 or not complainant_phone.strip().isdigit():
        raise HTTPException(status_code=400, detail="Complainant phone must be exactly 10 digits")
    if mobile.strip() and (len(mobile.strip()) != 10 or not mobile.strip().isdigit()):
        raise HTTPException(status_code=400, detail="Mobile number must be exactly 10 digits")
    if aadhaar.strip() and (len(aadhaar.strip()) != 12 or not aadhaar.strip().isdigit()):
        raise HTTPException(status_code=400, detail="Aadhaar must be exactly 12 digits")
    image_bytes = await image.read()
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be 8 MB or smaller")
    try:
        faces = extract_faces(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {exc}")
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    if face_index < 0 or face_index >= len(faces):
        raise HTTPException(status_code=400, detail="Invalid face selection")
    case_id = secrets.token_hex(16)
    case = RegisteredCase(
        id=case_id,
        submitted_by=username,
        name=name.strip(),
        father_name=father_name.strip() or None,
        age=age.strip() or None,
        complainant_name=complainant_name.strip(),
        complainant_mobile=complainant_phone.strip(),
        complainant_email=complainant_email.strip() or None,
        adhaar_card=aadhaar.strip() or None,
        last_seen=last_seen.strip(),
        address=address.strip() or None,
        city=city.strip() or None,
        description=description.strip() or None,
        face_mesh=json.dumps(faces[face_index]["landmarks"]),
        image_data=image_bytes,
    )
    with Session(engine) as session:
        session.add(case)
        session.commit()
    return {"ok": True, "id": case_id, "faces_detected": len(faces)}


@app.post("/api/sightings")
async def create_sighting(
    name: str = Form(...),
    mobile: str = Form(...),
    email: str = Form(""),
    location: str = Form(...),
    birth_marks: str = Form(""),
    image: UploadFile = File(...),
):
    if len(mobile.strip()) != 10 or not mobile.strip().isdigit():
        raise HTTPException(status_code=400, detail="Mobile number must be exactly 10 digits")
    image_bytes = await image.read()
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be 8 MB or smaller")
    try:
        faces = extract_faces(image_bytes, max_faces=1)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {exc}")
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    sub_id = secrets.token_hex(16)
    row = PublicSubmission(
        id=sub_id,
        submitted_by=name.strip(),
        location=location.strip(),
        email=email.strip() or None,
        face_mesh=json.dumps(faces[0]["landmarks"]),
        mobile=mobile.strip(),
        birth_marks=birth_marks.strip() or None,
        image_data=image_bytes,
    )
    with Session(engine) as session:
        session.add(row)
        session.commit()
    return {"ok": True, "id": sub_id}


@app.get("/api/cases")
def list_cases(user=Depends(current_user)):
    username, profile = user
    is_admin = profile.get("role", "").lower() == "admin"
    with Session(engine) as session:
        query = select(RegisteredCase).order_by(RegisteredCase.submitted_on.desc())
        if not is_admin:
            query = query.where(RegisteredCase.submitted_by == username)
        rows = session.exec(query).all()
    return [{
        "id": r.id, "name": r.name, "age": r.age, "status": r.status,
        "last_seen": r.last_seen, "city": r.city, "submitted_by": r.submitted_by,
        "matched_with": r.matched_with, "submitted_on": r.submitted_on.isoformat()
    } for r in rows]


@app.get("/api/public-cases")
def list_public_cases(user=Depends(current_user)):
    username, profile = user
    if profile.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    with Session(engine) as session:
        rows = session.exec(select(PublicSubmission).order_by(PublicSubmission.submitted_on.desc())).all()
    return [{
        "id": r.id, "status": r.status, "location": r.location,
        "mobile": r.mobile, "birth_marks": r.birth_marks,
        "submitted_by": r.submitted_by, "submitted_on": r.submitted_on.isoformat()
    } for r in rows]


@app.get("/api/cases/{case_id}/image")
def case_image(case_id: str, user=Depends(current_user)):
    username, profile = user
    with Session(engine) as session:
        row = session.get(RegisteredCase, case_id)
    if not row or (profile.get("role", "").lower() != "admin" and row.submitted_by != username):
        raise HTTPException(status_code=404, detail="Case not found")
    if not row.image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    from fastapi.responses import Response
    return Response(content=row.image_data, media_type="image/jpeg")


@app.get("/api/dashboard")
def dashboard(user=Depends(current_user)):
    username, profile = user
    is_admin = profile.get("role", "").lower() == "admin"
    with Session(engine) as session:
        q = select(RegisteredCase)
        if not is_admin:
            q = q.where(RegisteredCase.submitted_by == username)
        rows = session.exec(q).all()
    found = sum(r.status == "F" for r in rows)
    not_found = sum(r.status != "F" for r in rows)
    cities = {}
    for r in rows:
        city = r.city or "Unknown"
        d = cities.setdefault(city, {"found": 0, "not_found": 0})
        d["found" if r.status == "F" else "not_found"] += 1
    return {"found": found, "not_found": not_found, "total": len(rows), "cities": cities}


def _distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if denom == 0 else float(1.0 - np.dot(a, b) / denom)


@app.post("/api/match")
def match_cases(data: MatchRequest, user=Depends(current_user)):
    username, profile = user
    if profile.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="Only Admins can trigger matching")
    threshold = float(data.threshold)
    with Session(engine) as session:
        registered = session.exec(
            select(RegisteredCase).where(RegisteredCase.status == "NF")
        ).all()
        public = session.exec(
            select(PublicSubmission).where(PublicSubmission.status == "NF")
        ).all()
    if not registered or not public:
        return {"ok": True, "matches": [], "message": "No public or registered cases found"}

    X, labels = [], []
    for r in registered:
        try:
            X.append(json.loads(r.face_mesh))
            labels.append(r.id)
        except Exception:
            continue
    if not X:
        return {"ok": True, "matches": []}

    knn = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree", weights="distance")
    knn.fit(np.asarray(X, dtype=float), list(range(len(labels))))

    matches = []
    for p in public:
        try:
            vec = np.asarray(json.loads(p.face_mesh), dtype=float)
            distance = float(knn.kneighbors([vec])[0][0][0])
            if distance <= threshold:
                idx = int(knn.predict([vec])[0])
                case_id = labels[idx]
                with Session(engine) as session:
                    case = session.get(RegisteredCase, case_id)
                    sighting = session.get(PublicSubmission, p.id)
                    if case and sighting and case.status == "NF" and sighting.status == "NF":
                        case.status = "F"
                        case.matched_with = sighting.id
                        sighting.status = "F"
                        session.add(case)
                        session.add(sighting)
                        session.commit()
                confidence = max(0.0, min(100.0, (1.0 - distance / threshold) * 100.0))
                matches.append({
                    "case_id": case_id, "sighting_id": p.id,
                    "distance": distance, "confidence": confidence,
                })
        except Exception:
            continue
    return {"ok": True, "matches": matches}


@app.get("/api/cities")
def cities(user=Depends(current_user)):
    data = dashboard(user)
    return data["cities"]


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{path:path}")
def spa(path: str):
    # API routes are defined above; remaining routes serve the single-page UI.
    if path.startswith("api/") or path.startswith("static/"):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(FRONTEND_DIR / "index.html")
