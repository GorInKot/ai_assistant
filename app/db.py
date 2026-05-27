from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

DATABASE_URL = "sqlite:///./app_data.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Базовые роли — сидим в init_db().
ROLE_ADMIN = "admin"
ROLE_MANAGER = "manager"
ROLE_USER = "user"

DEFAULT_ROLES: tuple[tuple[str, str], ...] = (
    (ROLE_ADMIN, "Полный доступ ко всем разделам и админке"),
    (ROLE_MANAGER, "Управление сотрудниками и областями ответственности"),
    (ROLE_USER, "Обычный пользователь чата"),
)

# Базовые области ответственности — сидим один раз для удобства старта.
DEFAULT_RESPONSIBILITY_AREAS: tuple[tuple[str, str, str], ...] = (
    ("training", "Обучение", "Запись на курсы, повышение квалификации"),
    ("medical", "Медосмотр", "Периодические и предварительные медосмотры"),
    ("ektp", "ЕКТП / Транспорт", "Заявки на служебный транспорт"),
    ("tsus", "ЦУС / Стройконтроль", "Доступы и работа в ЦУС"),
    ("ai_lab", "Лаборатория ИИ", "Заявки в лабораторию ИИ"),
    ("legal", "Юридические вопросы / Законы РФ", "Консультации по трудовому праву и иным законам"),
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
    no_exact_match = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


class Employee(Base):
    """Сотрудник компании.

    user_id — опциональная связь с учётной записью в системе (если сотрудник
    залогинен — может быть автоматически назначен на связанные обращения).
    Email уникальный — служит первичным ключом при импорте из Excel.
    """

    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=False)
    position = Column(String, nullable=True)
    division = Column(String, nullable=True, index=True)
    subdivision = Column(String, nullable=True, index=True)
    phone = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    responsibilities = relationship(
        "Responsibility",
        back_populates="employee",
        cascade="all, delete-orphan",
    )


class ResponsibilityArea(Base):
    """Область ответственности: 'обучение', 'медосмотр', 'ЕКТП', и т.п.

    slug используется в каталоге типов заявок (см. Этап 5) и API-поиске.
    """

    __tablename__ = "responsibility_areas"
    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)

    responsibilities = relationship(
        "Responsibility",
        back_populates="area",
        cascade="all, delete-orphan",
    )


class Responsibility(Base):
    """Связь: сотрудник X отвечает за область Y в области действия Z.

    scope_division/subdivision = NULL означает "по всей компании" (fallback).
    is_primary помечает основного ответственного (а не зама).
    """

    __tablename__ = "responsibilities"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    area_id = Column(Integer, ForeignKey("responsibility_areas.id", ondelete="CASCADE"), nullable=False)
    scope_division = Column(String, nullable=True, index=True)
    scope_subdivision = Column(String, nullable=True, index=True)
    is_primary = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "employee_id", "area_id", "scope_division", "scope_subdivision",
            name="uq_responsibility_scope",
        ),
    )

    employee = relationship("Employee", back_populates="responsibilities")
    area = relationship("ResponsibilityArea", back_populates="responsibilities")


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_user_table()
    _seed_roles()
    _seed_responsibility_areas()
    _backfill_admin_role()


def _migrate_user_table():
    with engine.connect() as connection:
        existing_columns = set()
        result = connection.execute(text("PRAGMA table_info(users)"))
        for row in result:
            existing_columns.add(row[1])

        for column_name, column_type in (
            ("first_name", "TEXT"),
            ("last_name", "TEXT"),
            ("middle_name", "TEXT"),
            ("division", "TEXT"),
            ("subdivision", "TEXT"),
            ("job_title", "TEXT"),
        ):
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}"))


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


def _seed_responsibility_areas():
    db = SessionLocal()
    try:
        for slug, name, description in DEFAULT_RESPONSIBILITY_AREAS:
            existing = db.query(ResponsibilityArea).filter(ResponsibilityArea.slug == slug).first()
            if not existing:
                db.add(ResponsibilityArea(slug=slug, name=name, description=description))
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
