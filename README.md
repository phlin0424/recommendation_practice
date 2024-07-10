# MLflow Docker Setup

This setup involves two Docker containers: one for the MLflow server and one for the pipeline, both using the `/app/mlflow/artifacts` directory for artifacts. The MLflow server container runs with the command `mlflow server --backend-store-uri sqlite:///app/mlflow/mlflow.db --default-artifact-root /app/mlflow/artifacts --host 0.0.0.0 --port 5000`. The pipeline container logs artifacts using a relative path.

## System Architecture

```mermaid
graph LR
    A[Host Machine] 
    subgraph MLflow Server Container
        direction TB
        A1[/app/mlflow/artifacts/] -- Mounts to --> A2[/app/mlflow/artifacts/]
        A3[/app/mlflow/mlflow.db/] -- Mounts to --> A4[/app/mlflow/mlflow.db/]
        A5 --> |mlflow server --backend-store-uri sqlite:///app/mlflow/mlflow.db --default-artifact-root /app/mlflow/artifacts --host 0.0.0.0 --port 5000| A2
    end

    subgraph Pipeline Container
        direction TB
        B1[/app/mlflow/artifacts/] -- Mounts to --> B2[/app/mlflow/artifacts/]
        B3 --> |joblib.dump() + mlflow.log_artifact()| B2
    end

    A --> |Shared Volume| A2
    A --> |Shared Volume| B2
```