"""Alembic environment.

Источник истины по схеме — модели в app/db.py (Base.metadata). URL подключения
берём из того же DATABASE_URL, что и приложение (sqlite локально, postgres на
проде), а не из alembic.ini — чтобы миграции всегда шли в ту же БД.

render_as_batch=True обязателен для SQLite: ALTER TABLE там урезан, и Alembic
эмулирует изменения через пересоздание таблицы («batch» режим).
"""

from sqlalchemy import engine_from_config, pool

from alembic import context

# Метаданные приложения — для autogenerate и сверки схемы.
from app.db import DATABASE_URL, Base

config = context.config

# Намеренно НЕ вызываем fileConfig(alembic.ini): init_db() запускает миграции
# программно при старте приложения, а fileConfig с disable_existing_loggers=True
# отключил бы логгеры приложения и приглушил root до WARN. Логирование остаётся
# таким, каким его настроило приложение.

# URL берём из окружения приложения (перекрывает заглушку в alembic.ini).
config.set_main_option("sqlalchemy.url", DATABASE_URL)

target_metadata = Base.metadata


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=_is_sqlite(url or ""),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=_is_sqlite(str(connection.engine.url)),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
