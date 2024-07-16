from datetime import datetime
from datareader.db import fetch_datas
from pydantic import BaseModel
from datareader.ml_data_base import AbstractDatas
from table_models.ml_10m.settings import Ratings as Ratings_model
from table_models.ml_10m.settings import Movies as Movies_model
from table_models.ml_10m.settings import Tags as Tags_model


class Rating(BaseModel):
    user_id: int
    movie_id: int
    rating: int
    timestamp: datetime


class Movie(BaseModel):
    movie_id: int
    title: str
    genres: list[str]


class Tag(BaseModel):
    id: int
    user_id: int
    movie_id: int
    tag: str
    timestamp: datetime


class Ratings(AbstractDatas):
    data: list[Rating]

    @classmethod
    async def from_db(cls) -> "Ratings":
        ratings = await fetch_datas(Ratings_model)
        read_data = [
            Rating(
                user_id=rating.user_id,
                movie_id=rating.movie_id,
                rating=rating.rating,
                timestamp=rating.timestamp,
            )
            for rating in ratings
        ]
        return cls(data=read_data)


class Movies(AbstractDatas):
    data: list[Movie]

    @classmethod
    async def from_db(cls) -> "Movies":
        movies = await fetch_datas(Movies_model)
        read_data = [
            Movie(
                movie_id=movie.movie_id,
                title=movie.title,
                genres=movie.genres.split("|"),
            )
            for movie in movies
        ]
        return cls(data=read_data)


class Tags(AbstractDatas):
    data: list[Tag]

    @classmethod
    async def from_db(cls) -> "Tags":
        tags = await fetch_datas(Tags_model)
        # Read the tag in lower case
        read_data = [
            Tag(
                id=tag.id,
                user_id=tag.user_id,
                movie_id=tag.movie_id,
                tag=tag.tag.lower(),
                timestamp=tag.timestamp,
            )
            for tag in tags
        ]
        return cls(data=read_data)


if __name__ == "__main__":
    import asyncio
    import time

    # async def _main():
    #     ratings = await Ratings.from_db()
    #     train_data, test_data = ratings.split_data()
    #     print(test_data[0])
    #     print(test_data[1])
    #     print(len(ratings.data))
    #     print(len(test_data))
    #     print(len(train_data))

    async def _main():
        movies = await Movies.from_db()
        train_data, test_data = movies.split_data()
        print(movies.data[0])

        print(test_data[0])
        print(test_data[1])
        print("length of all data: ", len(movies.data))
        print(len(test_data))
        print(len(train_data))

    print("loading movies data")
    start_time = time.time()
    asyncio.run(_main())
    end_time = time.time()
    print(end_time - start_time)
