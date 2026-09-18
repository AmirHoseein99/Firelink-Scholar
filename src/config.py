from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = "openrouter"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""
    llm_proxy: str = ""
    groq_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project_name: str = "Firelink-Scholar"
    langsmith_run_name: str = "Firelink-Scholar-Run"


settings = Settings()
