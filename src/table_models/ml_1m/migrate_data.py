from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from table_models.ml_1m.settings import SCHEMA_NAME, Base

# Database connection string
DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost/pgvector_db"

# Create an engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Create the tables in the specified schema
Base.metadata.create_all(engine)


# Path to load the data
# TODO: integrate to core.config
data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ml-1m"


users = pd.read_csv(
    data_dir / "users.dat",
    sep="::",
    header=None,
    names=["user_id", "gender", "age", "occupation", "zipcode"],
    engine="python",
    encoding="ISO-8859-1",
)

movies = pd.read_csv(
    data_dir / "movies.dat",
    sep="::",
    header=None,
    names=["movie_id", "title", "genres"],
    engine="python",
    encoding="ISO-8859-1",
)

ratings = pd.read_csv(
    data_dir / "ratings.dat",
    sep="::",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
    engine="python",
    encoding="ISO-8859-1",
)

# Convert the Timestamp column to datetime
ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")


def migrate_to_ml_1m():
    # Insert the data into the database
    users.to_sql(
        "users",
        engine,
        if_exists="append",
        index=False,
        schema=SCHEMA_NAME,
    )
    movies.to_sql(
        "movies",
        engine,
        if_exists="append",
        index=False,
        schema=SCHEMA_NAME,
    )
    ratings.to_sql(
        "ratings",
        engine,
        if_exists="append",
        index=False,
        schema=SCHEMA_NAME,
    )


if __name__ == "__main__":
    # Insert the data into the database
    migrate_to_ml_1m()
