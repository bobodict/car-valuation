from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus
import os

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
REPOSITORY_DIR = PROJECT_DIR.parent

for env_path in (BACKEND_DIR / ".env", REPOSITORY_DIR / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _build_database_url() -> str:
    configured_url = os.getenv("DATABASE_URL")
    if configured_url:
        return configured_url

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME")
    if user and password and name:
        return (
            f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{quote_plus(name)}?charset=utf8mb4"
        )

    sqlite_path = (BACKEND_DIR / "car_valuation.db").as_posix()
    return f"sqlite:///{sqlite_path}"


@dataclass(frozen=True)
class Settings:
    database_url: str
    allowed_origins: tuple[str, ...]
    models_dir: Path = BACKEND_DIR / "models"

    @property
    def preprocess_path(self) -> Path:
        return self.models_dir / "preprocess.joblib"

    @property
    def feature_config_path(self) -> Path:
        return self.models_dir / "feature_config.json"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "price_mlp.pt"

    @property
    def metrics_path(self) -> Path:
        return self.models_dir / "metrics.json"


def _allowed_origins() -> tuple[str, ...]:
    raw = os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


settings = Settings(
    database_url=_build_database_url(),
    allowed_origins=_allowed_origins(),
)
