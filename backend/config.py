from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus
import os

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

for env_path in (BACKEND_DIR / ".env", PROJECT_DIR / ".env"):
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
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: float
    models_dir: Path = BACKEND_DIR / "models"

    @property
    def experiment_path(self) -> Path:
        return self.models_dir

    @property
    def manifest_path(self) -> Path:
        return self.experiment_path / "model_manifest.json"

    @property
    def model_manifest_path(self) -> Path:
        return self.manifest_path

    @property
    def preprocess_path(self) -> Path:
        return self.experiment_path / "preprocess.joblib"

    @property
    def feature_config_path(self) -> Path:
        return self.experiment_path / "feature_config.json"

    @property
    def model_path(self) -> Path:
        return self.experiment_path / "price_mlp.pt"

    @property
    def metrics_path(self) -> Path:
        return self.experiment_path / "metrics.json"

    @property
    def knowledge_path(self) -> Path:
        return BACKEND_DIR / "data" / "knowledge_base.json"


def _allowed_origins() -> tuple[str, ...]:
    raw = os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


settings = Settings(
    database_url=_build_database_url(),
    allowed_origins=_allowed_origins(),
    llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
    llm_api_key=os.getenv("LLM_API_KEY", ""),
    llm_model=os.getenv("LLM_MODEL", ""),
    llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
)
