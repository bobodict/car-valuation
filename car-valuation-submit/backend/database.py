# database.py
from sqlalchemy import create_engine
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
    """Add non-destructive model metadata columns to an existing SQLite table."""
    if db_engine.dialect.name != "sqlite":
        return
    with db_engine.begin() as connection:
        columns = {
            row[1]
            for row in connection.exec_driver_sql("PRAGMA table_info(history)").fetchall()
        }
        for column_name, column_type in (
            ("currency", "VARCHAR(16)"),
            ("model_version", "VARCHAR(64)"),
        ):
            if column_name not in columns:
                connection.exec_driver_sql(
                    f'ALTER TABLE history ADD COLUMN "{column_name}" {column_type}'
                )
