# MLflow-Based Machine Learning Experimentation for Recommendation Systems


This repository facilitates the experimentation and evaluation of machine learning models, primarily for recommendation systems using MLflow for tracking experiments. 

The project is structured around Docker containers, ensuring smooth orchestration and execution of model pipelines.



## Key Features

This repository contains:


- **MLflow for Experiment Tracking:** Used to track machine learning experiments, log parameters, metrics, and store artifacts.

- **FTP Server for Artifact Storage:** Artifacts, such as model outputs, are stored on an FTP server.

- **PostgreSQL Database:** The MovieLens dataset (`ml-10m`) is migrated into a PostgreSQL database using Alembic migrations.


- **Machine Learning Model Pipelines:** Each machine learning model is structured as a pipeline. Pipelines read data from the database using SQL queries and SQLAlchemy, process the data, train models, and evaluate performance. Each pipeline is executed within its own Docker container, ensuring modularity and easy scaling. 


- **Docker Compose:** All containers (MLflow, FTP server, PostgreSQL, and pipeline execution) are orchestrated using Docker Compose for simplified management.


## Project Structure


### Folder Structure

The main code is under the `src/` directory, which contains the following modules:
- `alembic/`: Code to execute migrations. 
- `core/`: Loads environment variables using Pydantic settings.
- `datareader/`: Contains modules to read data from the database.
- `pipelines/`: Contains pipelines and models for different machine learning experiments. Each pipeline consists of the following components:
    
    - `Dockerfile`: Used to launch the container for the pipeline.
    - `model.py`: Contains the code for the machine learning model.
    - `pipeline_{model_name}.py`: Contains the code for the specific pipeline corresponding to the model.
    - `pyproject.toml`: Defines the environment and dependencies for the container.
    - `pipeline_params.env`: Stores the environment variables used in the pipeline.


- `sql/`: Stores SQL queries used for data extraction and transformation.
- `table_models/`: Defines database models using SQLAlchemy. Also include the code to execute the migration. 
- `utils/`: Contains various helper functions to support other modules.


### Additional Files 

- `Dockerfile`: Defines the instructions to build Docker images for running the pipelines, MLflow, and other services. This file describes the environment setup, including the installation of dependencies.
- `docker-compose.yml`: Orchestrates multiple Docker containers (MLflow, PostgreSQL, FTP, and pipelines) to run the entire system seamlessly.
- `poetry.lock` / `pyproject.toml`: Lists the Python dependencies needed for the project.
- `alembic.ini`: Handles the database migrations for loading the MovieLens dataset into PostgreSQL.
- `.env`: Stores environment variables such as database credentials and configuration settings.


## How to Run


### Prerequisites

tbc

### Steps to run the system

1. Clone this repository:

```shell 
git clone <repository-url>
```

2. Start all necessary services (MLflow, PostgreSQL, FTP, etc.) using Docker Compose:
```
docker-compose up
```

3. Execute migration: 

```shell
# Assume that you have installed poetry
poetry install

# Create the initial tables for the dataset
alembic revision --autogenerate -m "Create initial tables"

# Populate the dataset to DB
poetry run python table_models/ml_10m/migrate_data.py
```


4. To run a machine learning pipeline (train and evaluate a model), use:

```shell
# Run the SVD recommender pipeline for example: 
docker-compose --profile pipeline_svd up
```

### Steps to Configure `.env`

tbc