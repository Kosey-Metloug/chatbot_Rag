from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://app:secret@localhost:5432/docs"
    chat_model: str = "qwen3.5:2b"
    embed_model: str = "nomic-embed-text"
    docs_dir: str = "./pdfs"
    top_k: int = 3


settings = Settings()
