from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import mlflow

DIR_PATH = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    git_python_refresh: str
    mlflow_tracking_host: str
    mlflow_tracking_port: int
    db_host: str
    ftp_user_name: str
    ftp_user_pass: str
    ftp_user_home: str
    ftp_host: str
    postgres_user: str
    postgres_password: str
    postgres_db: str
    experiment_name: str
    backend_store_uri: str

    @property
    def artifact_location(self) -> str:
        return f"ftp://{self.ftp_user_name}:{self.ftp_user_pass}@{self.ftp_host}/"

    @property
    def tracking_uri(self) -> str:
        return f"http://{self.mlflow_tracking_host}:{self.mlflow_tracking_port}"

    @property
    def experiment_id(self) -> int:
        mlflow.set_tracking_uri(self.tracking_uri)
        # Ensure that the specified experiment name exist
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        return experiment.experiment_id

    model_config = SettingsConfigDict(
        env_file=DIR_PATH / ".env",
        env_file_encoding="utf-8",
    )


# Load all the environment variables
settings = Settings()

if __name__ == "__main__":
    print(f"Loading .env file from: {DIR_PATH } /  .env")
    print(f"Tracking URI: {settings.tracking_uri}")
    print(f"DB Host: {settings.db_host}")
    print(settings.model_dump())  # Print all loaded settings
