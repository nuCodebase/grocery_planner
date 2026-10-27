import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class Config:
    """Application configuration loaded from environment variables and .env."""

    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parent
        self._load_dotenv()
        self.pdf_src = self._resolve_path(
            os.getenv("PDF_SRC"), Path("ingestion/whirlpool_oven_manual2.pdf")
        )
        self.parsed_output = self._resolve_path(
            os.getenv("PARSED_OUTPUT"), Path("ingestion/parsed_output")
        )
        self.parsed_output.mkdir(exist_ok=True, parents=True)
        self.chunking_lib = os.getenv("CHUNKING_LIB", "langchain")
        self.chunking_type = os.getenv("CHUNKING_TYPE", "markdown")

    def _load_dotenv(self) -> None:
        if load_dotenv is not None:
            load_dotenv(dotenv_path=self.repo_root / ".env")

    def _resolve_path(self, env_value: str | None, default_path: Path) -> Path:
        path = Path(env_value) if env_value else default_path
        if path.is_absolute():
            return path
        return (self.repo_root / path).resolve()
    