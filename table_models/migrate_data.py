import pandas as pd
from table_models.settings import SCHEMA_NAME, Base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from pathlib import Path

# Database connection string
DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost/pgvector_db"

# Create an engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Create the tables in the specified schema
Base.metadata.create_all(engine)


# Path to load the data
data_dir = Path(__file__).resolve().parent.parent / "data" / "ml-1m"


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


if __name__ == "__main__":
    # Insert the data into the database
    users.to_sql("users", engine, if_exists="append", index=False, schema=SCHEMA_NAME)
    movies.to_sql("movies", engine, if_exists="append", index=False, schema=SCHEMA_NAME)
    ratings.to_sql(
        "ratings", engine, if_exists="append", index=False, schema=SCHEMA_NAME
    )
