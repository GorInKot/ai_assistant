from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./app_data.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Модель пользователя
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
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Модель действий пользователя (можно расширить)
class UserAction(Base):
    __tablename__ = "user_actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    action_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    details = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_user_table()


def _migrate_user_table():
    existing_columns = set()
    with engine.connect() as connection:
        result = connection.execute(text("PRAGMA table_info(users)"))
        for row in result:
            existing_columns.add(row[1])

        for column_name, column_type in (
            ("first_name", "TEXT"),
            ("last_name", "TEXT"),
            ("middle_name", "TEXT"),
            ("division", "TEXT"),
            ("subdivision", "TEXT"),
        ):
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}"))
