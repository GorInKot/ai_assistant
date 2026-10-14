import os

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# DATABASE_URL env-driven: для prod на Render с persistent disk можно
# поставить sqlite:////data/app_data.db, для Postgres — postgresql://...
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app_data.db")

connect_args: dict = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Базовые роли — сидим в init_db().
ROLE_ADMIN = "admin"
ROLE_MANAGER = "manager"
ROLE_USER = "user"

DEFAULT_ROLES: tuple[tuple[str, str], ...] = (
    (ROLE_ADMIN, "Полный доступ ко всем разделам и админке"),
    (ROLE_MANAGER, "Управление базой знаний: загрузка документов и переиндексация"),
    (ROLE_USER, "Обычный пользователь чата"),
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    middle_name = Column(String, nullable=True)
    division = Column(String, nullable=True)
    subdivision = Column(String, nullable=True)
    job_title = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    roles = relationship(
        "Role",
        secondary="user_roles",
        back_populates="users",
        lazy="joined",
    )


class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(String, nullable=True)

    users = relationship("User", secondary="user_roles", back_populates="roles")


class UserRole(Base):
    """M:M между users и roles. Отдельная таблица позволяет позже добавить
    created_at / granted_by без миграции колонки."""

    __tablename__ = "user_roles"
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)


class UserAction(Base):
    __tablename__ = "user_actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    action_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    details = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=False, default="Новая беседа")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)
    # Проверка обоснованности ответа (цитаты, неподтверждённые утверждения) — JSON или NULL.
    grounding_json = Column(Text, nullable=True)
    no_exact_match = Column(Integer, nullable=False, default=0)
    # Оценка ответа пользователем: 1 — полезно, -1 — не полезно, NULL — не оценён.
    rating = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


class QaLog(Base):
    """Журнал вопросов к базе знаний — основа аналитики пробелов (services/gaps.py).

    Живёт отдельно от `messages`: беседы пользователь может удалить, а статистика
    должна остаться. `message_id` — ссылка на ответ без внешнего ключа: связь нужна лишь
    пока беседа существует, чтобы перенести оценку в журнал.
    """

    __tablename__ = "qa_log"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    message_id = Column(Integer, nullable=True, index=True)
    # Нужен, чтобы считать разных людей, а не повторы одного и того же вопроса. Без внешнего
    # ключа; отчёт отдаёт только числа, кто именно спрашивал — не показывается.
    user_id = Column(Integer, nullable=True)
    # Вопрос с учётом уточнения («по ЕКТП» дописывается к исходному вопросу).
    question = Column(Text, nullable=False)
    # answer — ответ по документам; missing — в базе не нашлось; clarify — просьба уточнить.
    kind = Column(String, nullable=False)
    division = Column(String, nullable=True)
    # Проверка обоснованности: сколько утверждений в ответе и сколько без опоры в документах.
    claims_total = Column(Integer, nullable=True)
    claims_unsupported = Column(Integer, nullable=True)
    # JSON: [{"relative_path": ..., "source_type": ...}] — какие документы показаны к ответу.
    sources_json = Column(Text, nullable=True)
    rating = Column(Integer, nullable=True)


class CadRequest(Base):
    """История запросов во вкладке «Конструктор деталей» (на пользователя).

    Картинки и STEP не хранятся: деталь собирается заново из спецификации (`spec_json`) без
    обращения к модели, либо запрос повторяется по описанию.
    """

    __tablename__ = "cad_requests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    description = Column(Text, nullable=False)
    # built — деталь построена; clarify — модель попросила уточнить; unsupported — «пока не
    # поддерживается»; error — не удалось составить или построить деталь.
    outcome = Column(String, nullable=False)
    # part — деталь (3D, SolidCraft); route — чертёж трассы (2D, CADCraft).
    kind = Column(String, nullable=False, default="part", server_default="part")
    title = Column(String, nullable=True)
    # Вопрос на уточнение, текст «пока не поддерживается» или сообщение об ошибке.
    detail = Column(Text, nullable=True)
    spec_json = Column(Text, nullable=True)
    # JSON-список пометок («Материал не назван…», «Пока не поддерживается и не учтено: …»).
    notes_json = Column(Text, nullable=True)


def init_db():
    _run_migrations()
    _seed_roles()
    _seed_initial_admin()
    _backfill_admin_role()


def _alembic_config():
    """Config Alembic, привязанный к alembic.ini проекта и текущему DATABASE_URL."""
    import os

    from alembic.config import Config

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = Config(os.path.join(base_dir, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(base_dir, "alembic"))
    cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
    return cfg


def _run_migrations():
    """Приводит схему БД к head через Alembic (заменяет create_all + ручные ALTER).

    Три сценария:
    - alembic_version есть → обычный upgrade head (применить новые миграции).
    - alembic_version нет, но таблицы уже есть (БД создана старым create_all до
      внедрения Alembic) → принимаем существующую схему за baseline: stamp на
      базовую ревизию + upgrade head (применит миграции после baseline, если
      появятся). Делается один раз — дальше alembic_version уже будет.
    - alembic_version нет и таблиц нет (чистая БД) → upgrade head создаёт всё.
    """
    from alembic import command
    from alembic.script import ScriptDirectory
    from sqlalchemy import inspect

    tables = set(inspect(engine).get_table_names())
    cfg = _alembic_config()

    if "alembic_version" not in tables and "users" in tables:
        base_rev = ScriptDirectory.from_config(cfg).get_bases()[0]
        command.stamp(cfg, base_rev)

    command.upgrade(cfg, "head")


def _seed_roles():
    db = SessionLocal()
    try:
        for name, description in DEFAULT_ROLES:
            existing = db.query(Role).filter(Role.name == name).first()
            if not existing:
                db.add(Role(name=name, description=description))
        db.commit()
    finally:
        db.close()


def _seed_initial_admin():
    """Если в env заданы INITIAL_ADMIN_EMAIL + INITIAL_ADMIN_PASSWORD —
    гарантируем, что такой пользователь есть и у него есть роль admin.

    Зачем: на ephemeral-инфраструктуре (Render free plan) БД сбрасывается
    при каждом деплое/cold start. assign_initial_role даёт admin'а первому
    зарегистрировавшемуся — но это race condition: кто-то снаружи может
    успеть первым. INITIAL_ADMIN_* env закрывает дыру: при старте сразу
    есть admin, и assign_initial_role даст следующим только role=user.

    Идемпотентно: при наличии user с таким email и role admin ничего не
    делает; добавляет роль если её нет; пересоздаёт пароль НЕ трогает
    (если admin захочет ротировать пароль — через UI).
    """
    email = os.getenv("INITIAL_ADMIN_EMAIL")
    password = os.getenv("INITIAL_ADMIN_PASSWORD")
    if not email or not password:
        return

    # local import — passlib тяжёлый, и app/auth.py импортирует app/db.py.
    from passlib.context import CryptContext
    pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

    db = SessionLocal()
    try:
        admin_role = db.query(Role).filter(Role.name == ROLE_ADMIN).first()
        if not admin_role:
            return

        existing = db.query(User).filter(User.email == email).first()
        if existing:
            if not any(r.name == ROLE_ADMIN for r in existing.roles):
                existing.roles.append(admin_role)
                db.commit()
            return

        user = User(
            email=email,
            full_name="Initial Admin",
            first_name="Admin",
            last_name="Initial",
            hashed_password=pwd_ctx.hash(password),
        )
        user.roles.append(admin_role)
        # Также добавим базовую роль user, чтобы поведение совпадало с обычной
        # регистрацией (admin сам по себе подразумевает user).
        user_role = db.query(Role).filter(Role.name == ROLE_USER).first()
        if user_role:
            user.roles.append(user_role)
        db.add(user)
        db.commit()
    finally:
        db.close()


def _backfill_admin_role():
    """Если в системе нет ни одного admin, назначить им самого первого юзера.

    Нужно для миграции: до Этапа 4 ролей не было, и существующие пользователи
    остались без admin. Без этого никто не сможет открыть админ-эндпоинты.
    Идемпотентно: при наличии хотя бы одного admin ничего не делает.
    """
    db = SessionLocal()
    try:
        admin_role = db.query(Role).filter(Role.name == ROLE_ADMIN).first()
        if not admin_role:
            return

        any_admin = (
            db.query(User)
            .join(User.roles)
            .filter(Role.name == ROLE_ADMIN)
            .first()
        )
        if any_admin:
            return

        # Также назначим всем существующим пользователям без ролей роль user.
        user_role = db.query(Role).filter(Role.name == ROLE_USER).first()
        all_users = db.query(User).order_by(User.id).all()
        if not all_users:
            return

        for user in all_users:
            if not user.roles and user_role:
                user.roles.append(user_role)

        # Первый по id — становится admin (дополнительно к user).
        first = all_users[0]
        first.roles.append(admin_role)
        db.commit()
    finally:
        db.close()
