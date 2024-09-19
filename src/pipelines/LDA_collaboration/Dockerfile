FROM python:3.11-slim

WORKDIR  /app
RUN mkdir /app/src

# Install gcc and other necessary build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libpython3-dev

# Install poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Copy the poetry file to run the pipeline
COPY src/pipelines/LDA_collaboration/pyproject.toml /app/pyproject.toml

# Copy the module codes
COPY src /app/src

# Copy the pipeline script
COPY src/pipelines/LDA_collaboration/pipeline_lda_collaboration.py /app/src/pipeline_lda_collaboration.py

# Install the dependencies usnig poetry
RUN poetry config virtualenvs.in-project true
RUN poetry install --only main

# Use entrypoint script to run the appropriate pipeline
ENTRYPOINT [ "poetry", "run", "python", "src/pipeline_lda_collaboration.py"]
