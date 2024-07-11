from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import pandas as pd
from sqlalchemy import select
from table_models.settings import Ratings
import asyncio
import os

# TODO: integrate to core.config
DB_HOST = os.getenv("DB_HOST", "localhost")


# Define the database URL
DATABASE_URL = f"postgresql+asyncpg://postgres:postgres@{DB_HOST}/pgvector_db"

# Create an async engine and session
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Async function to fetch data from the database
async def fetch_ratings():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Ratings))
        ratings = result.scalars().all()
        return ratings


if __name__ == "__main__":
    # test that if we can fetch the data from db using async process:
    async def main():
        ratings = await fetch_ratings()

        ratings_data = [
            {
                "user_id": rating.user_id,
                "item_id": rating.item_id,
                "rating": rating.rating,
                "timestamp": rating.timestamp,
            }
            for rating in ratings
        ]
        print(pd.DataFrame(ratings_data))

    asyncio.run(main())
