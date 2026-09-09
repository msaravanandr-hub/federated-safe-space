import os
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from .database import get_db
from .models import User

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM, TOKEN_TTL = "HS256", timedelta(days=7)
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _pw_bytes(p: str) -> bytes:
    # bcrypt only uses the first 72 bytes; truncate explicitly for compatibility
    return p.encode("utf-8")[:72]


def hash_password(p: str) -> str:
    return bcrypt.hashpw(_pw_bytes(p), bcrypt.gensalt()).decode("utf-8")


def verify_password(p: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(_pw_bytes(p), h.encode("utf-8"))
    except ValueError:
        return False


def create_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "exp": datetime.now(timezone.utc) + TOKEN_TTL}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    err = HTTPException(401, "Invalid or expired token")
    try:
        uid = int(jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]).get("sub", 0))
    except (JWTError, ValueError):
        raise err
    user = db.get(User, uid)
    if user is None:
        raise err
    return user
