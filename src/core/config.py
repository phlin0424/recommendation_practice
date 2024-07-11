from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import mlflow

DIR_PATH = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    git_python_refresh: str
    mlflow_tracking_host: str
    mlflow_tracking_port: int
    db_host: str
    publichost: str
    ftp_user_name: str
    ftp_user_pass: str
    ftp_user_home: str
    postgres_user: str
    postgres_password: str
    postgres_db: str
    experiment_name: str
    backend_store_uri: str
    postgres_user: str
    postgres_password: str
    postgres_db: str

    @property
    def tracking_uri(self) -> str:
        return f"http://{self.mlflow_tracking_host}:{self.mlflow_tracking_port}"

    @property
    def experiment_id(self) -> int:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        return experiment.experiment_id

    model_config = SettingsConfigDict(
        env_file=DIR_PATH / ".env",
        env_file_encoding="utf-8",
    )


# Load all the environment variables
settings = Settings()

# http://mlflow-server:5000

if __name__ == "__main__":
    print(settings.tracking_uri)
