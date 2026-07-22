# database.py
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings


DATABASE_URL = settings.database_url
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=not DATABASE_URL.startswith("sqlite"),
    connect_args=connect_args,
)

# 会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ORM Base 类
Base = declarative_base()


# 提供依赖注入用的数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_history_metadata_columns(db_engine=engine):
    """Add non-destructive model metadata columns to an existing history table."""
    inspector = inspect(db_engine)
    if not inspector.has_table("history"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("history")}
    missing_columns = [
        (column_name, column_type)
        for column_name, column_type in (
            ("currency", "VARCHAR(16)"),
            ("model_version", "VARCHAR(64)"),
        )
        if column_name not in existing_columns
    ]
    if not missing_columns:
        return

    preparer = db_engine.dialect.identifier_preparer
    table_name = preparer.quote("history")
    with db_engine.begin() as connection:
        for column_name, column_type in missing_columns:
            quoted_column = preparer.quote(column_name)
            connection.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN {quoted_column} {column_type} NULL"
            )
