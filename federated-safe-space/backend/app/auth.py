import os
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
# from passlib.context import CryptContext
from pwdlib import PasswordHash
from sqlalchemy.orm import Session
from .database import get_db
from .models import User

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM, TOKEN_TTL = "HS256", timedelta(days=7)
pwd = PasswordHash.recommended()
# pwd = CryptContext(schemes=["bcrypt"])
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(p: str) -> str:
    return pwd.hash(p)


def verify_password(p: str, h: str) -> bool:
    return pwd.verify(p, h)


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
