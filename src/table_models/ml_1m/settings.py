from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


SCHEMA_NAME = "ml_1m"


# Create a table Model which inherit from Base
class Users(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": SCHEMA_NAME}

    user_id = Column(Integer, primary_key=True, autoincrement=False)
    gender = Column(String(1), nullable=False)
    age = Column(Integer, nullable=False)
    occupation = Column(Integer, nullable=False)
    zipcode = Column(String(10), nullable=False)


# Create a table Model which inherit from Base
class Movies(Base):
    __tablename__ = "movies"
    __table_args__ = {"schema": SCHEMA_NAME}

    movie_id = Column(Integer, primary_key=True, autoincrement=False)
    title = Column(String, nullable=False)
    genres = Column(String, nullable=False)


# Create a table Model which inherit from Base
class Ratings(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "item_id", name="ratings_pk"),
        {"schema": SCHEMA_NAME},
    )

    # ForeignKey Relationships:
    # The user_id and item_id columns are defined as foreign keys that
    # reference the users and movies tables, respectively.
    # This ensures referential integrity
    user_id = Column(
        Integer, ForeignKey(f"{SCHEMA_NAME}.users.user_id"), autoincrement=False
    )
    item_id = Column(
        Integer, ForeignKey(f"{SCHEMA_NAME}.movies.movie_id"), autoincrement=False
    )
    rating = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)
