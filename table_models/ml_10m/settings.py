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


SCHEMA_NAME = "ml_10m"


# Create a hidden table
class Users(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": SCHEMA_NAME}
    user_id = Column(Integer, primary_key=True, autoincrement=True)


# Create a table Model which inherit from Base
class Tags(Base):
    __tablename__ = "tags"
    __table_args__ = {"schema": SCHEMA_NAME}

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    movie_id = Column(Integer, nullable=False)
    tag = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)


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
        PrimaryKeyConstraint("user_id", "movie_id", name="ratings_pk"),
        {"schema": SCHEMA_NAME},
    )
    user_id = Column(
        Integer, ForeignKey(f"{SCHEMA_NAME}.users.user_id"), autoincrement=False
    )
    movie_id = Column(
        Integer, ForeignKey(f"{SCHEMA_NAME}.movies.movie_id"), autoincrement=False
    )
    rating = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)
