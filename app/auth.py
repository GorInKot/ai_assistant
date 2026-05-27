import os
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.db import ROLE_ADMIN, ROLE_USER, Role, User, SessionLocal
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Секрет подписи JWT. В проде ОБЯЗАТЕЛЬНО задать переменную окружения JWT_SECRET —
# иначе по дефолтному значению можно подделать токен любого пользователя.
SECRET_KEY = os.getenv("JWT_SECRET", "dev-insecure-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto"
)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Не удалось авторизовать пользователя",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


def user_has_role(user: User, role_name: str) -> bool:
    return any(role.name == role_name for role in user.roles)


def require_role(*allowed_roles: str):
    """Dependency-фабрика для защиты эндпоинтов ролями.

    Использование: `current_user: User = Depends(require_role("admin"))`.
    Пользователь должен иметь хотя бы одну из перечисленных ролей.
    """

    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if any(role.name in allowed_roles for role in current_user.roles):
            return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Требуется одна из ролей: {', '.join(allowed_roles)}",
        )

    return dependency


def assign_initial_role(db: Session, user: User) -> None:
    """При регистрации: первый пользователь становится admin, остальные — user.

    Делается в той же транзакции, что и создание user — вызвать ДО commit.
    """
    user_count = db.query(User).count()
    role_name = ROLE_ADMIN if user_count == 0 else ROLE_USER
    role = db.query(Role).filter(Role.name == role_name).first()
    if role:
        user.roles.append(role)
    else:
        logger.warning("Role %s not seeded — user %s will have no role", role_name, user.email)
