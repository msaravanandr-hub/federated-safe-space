import base64
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .auth import current_user, create_token, hash_password, verify_password
from .models import User, Post, Report
from .federation.lagpsa import LagPsaAggregator, ClientUpdate
from .federation.model import MODEL_DIM, evaluate_global, featurize, load_test_set

N_PARAMS = MODEL_DIM  # hashed logistic-regression weights + bias
BASE_DIR = Path(__file__).resolve().parent
TEST_SET = load_test_set()
LAST_EVAL = {"round": 0, "auc": None, "f1": None, "n": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Postgres may still be starting when the container boots; retry until ready.
    for _ in range(30):
        try:
            Base.metadata.create_all(engine)
            break
        except Exception:
            time.sleep(2)
    yield


app = FastAPI(title="Federated Safe Space API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

agg = LagPsaAggregator(n_params=N_PARAMS, clip_norm=10.0, dp_sigma=1e-3, min_quorum=5, lr=1.0)

# ---------- auth ----------
class RegisterIn(BaseModel):
    username: str
    password: str
    consent_federated: bool = False


@app.post("/auth/register")
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(409, "Username taken")
    user = User(username=body.username, password_hash=hash_password(body.password),
                consent_federated=body.consent_federated)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "token": create_token(user.id)}


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == form.username))
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(401, "Bad credentials")
    return {"access_token": create_token(user.id), "token_type": "bearer"}


# ---------- posts & moderation ----------
class PostIn(BaseModel):
    body: str


BANNED = {"hate", "kill", "scam"}  # demo placeholder; the federated model replaces this


def moderation_score(text: str) -> float:
    return min(1.0, 0.25 * sum(w in text.lower() for w in BANNED))


def global_score(text: str) -> float:
    """Federated global model score once round 1 has aggregated;
    keyword bootstrap before that (the model literally does not exist yet)."""
    if agg.round_id == 0:
        return moderation_score(text)
    z = float(agg.W @ featurize(text))
    return float(1.0 / (1.0 + math.exp(-z)))


@app.post("/posts")
def create_post(body: PostIn, user=Depends(current_user), db: Session = Depends(get_db)):
    score = global_score(body.body)
    post = Post(author_id=user.id, body=body.body, toxicity_score=score)
    db.add(post)
    db.commit()
    db.refresh(post)
    return {"id": post.id, "toxicity_score": score, "flagged": score >= 0.5}


@app.get("/posts")
def list_posts(db: Session = Depends(get_db), limit: int = 50):
    rows = db.scalars(select(Post).order_by(Post.id.desc()).limit(min(limit, 100))).all()
    return [{"id": p.id, "author_id": p.author_id, "body": p.body,
             "toxicity_score": p.toxicity_score,
             "created_at": p.created_at.isoformat() if p.created_at else None} for p in rows]


class ReportIn(BaseModel):
    post_id: int
    reason: str


@app.post("/reports")
def create_report(body: ReportIn, user=Depends(current_user), db: Session = Depends(get_db)):
    if not db.get(Post, body.post_id):
        raise HTTPException(404, "Post not found")
    r = Report(post_id=body.post_id, reporter_id=user.id, reason=body.reason)
    db.add(r)
    db.commit()
    db.refresh(r)
    return {"id": r.id, "status": r.status}


# ---------- federation (LAG-PSA) ----------
class UpdateIn(BaseModel):
    client_id: str
    round_id: int
    staleness: int
    n_samples: int
    indices: list[int]
    values: list[float]


@app.post("/federation/upload-update")
def upload_update(body: UpdateIn, user=Depends(current_user)):
    u = ClientUpdate(client_id=body.client_id, round_id=body.round_id,
                     staleness=body.staleness, n_samples=body.n_samples,
                     indices=np.asarray(body.indices, dtype=np.int64),
                     values=np.asarray(body.values, dtype=np.float32))
    accepted = agg.accept(u)
    result = agg.maybe_aggregate() or {}
    if result:
        global LAST_EVAL
        LAST_EVAL = evaluate_global(agg.W, TEST_SET, result["round"])
        result["eval"] = LAST_EVAL
    return {"accepted": accepted, "aggregated": result}


@app.get("/federation/global-model")
def global_model():
    return {"round": agg.round_id, "n_params": agg.n_params,
            "pending_updates": len(agg.pending)}


@app.get("/federation/global-weights")
def global_weights():
    """Current global model weights (base64 float32). Clients pull this to train locally."""
    return {"round": agg.round_id,
            "weights_b64": base64.b64encode(agg.W.astype(np.float32).tobytes()).decode("ascii")}


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    return {"users": db.scalar(select(func.count(User.id))),
            "posts": db.scalar(select(func.count(Post.id))),
            "reports": db.scalar(select(func.count(Report.id))),
            "federation_round": agg.round_id,
            "eval": LAST_EVAL}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def dashboard():
    """Web dashboard UI (served by the API itself; no separate server needed)."""
    return FileResponse(BASE_DIR / "static" / "dashboard.html")
