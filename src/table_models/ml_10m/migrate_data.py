import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from table_models.ml_10m.settings import SCHEMA_NAME, Base

# Set up log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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
data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ml-10m"


def read_csv(file_name: Path, names: list[str]):
    logger.info(f"read csv: {file_name.name}")
    return pd.read_csv(
        file_name,
        sep="::",
        header=None,
        names=names,
        engine="python",
        encoding="ISO-8859-1",
    )


def create_users_df(df_ratings: pd.DataFrame, df_tags: pd.DataFrame):
    """
    Create a user_id table (which is a latent table)
    with the id ranging from 1 to the maximum user_id in other tables
    """
    user_id_from_df_ratings = df_ratings["user_id"].max()
    user_id_from_df_tags = df_tags["user_id"].max()

    user_id_range = max(user_id_from_df_ratings, user_id_from_df_tags)

    return pd.DataFrame({"user_id": list(range(1, user_id_range + 1))})


def migrate():
    ratings = read_csv(
        file_name=data_dir / "ratings.dat",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    # Convert the Timestamp column to datetime
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    movies = read_csv(
        file_name=data_dir / "movies.dat",
        names=["movie_id", "title", "genres"],
    )

    tags = read_csv(
        file_name=data_dir / "tags.dat",
        names=["user_id", "movie_id", "tag", "timestamp"],
    )
    # fix tag dataframe
    tags["timestamp"] = pd.to_datetime(tags["timestamp"], unit="s")
    tags.loc[tags["tag"].isnull(), "tag"] = "no_tag"

    users = create_users_df(ratings, tags)

    # Insert the data into the database
    logger.info("migrating: users")
    users.to_sql(
        "users",
        engine,
        if_exists="append",
        index=False,
        schema=SCHEMA_NAME,
    )

    logger.info("migrating: tags")
    tags.to_sql(
        "tags",
        engine,
        if_exists="append",
        index=False,
        schema=SCHEMA_NAME,
    )

    logger.info("migrating: movies")
    movies.to_sql(
        "movies",
        engine,
        if_exists="append",
        index=False,
        schema=SCHEMA_NAME,
    )

    logger.info("migrating: ratings")
    ratings.to_sql(
        "ratings",
        engine,
        if_exists="append",
        index=False,
        schema=SCHEMA_NAME,
    )


if __name__ == "__main__":
    # Insert the data into the database
    migrate()
