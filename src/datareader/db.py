import asyncio

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from table_models.ml_1m.settings import Ratings
from core.config import settings

DB_HOST = settings.db_host
POSTGRES_DB = settings.postgres_db
POSTGRES_PASSWORD = settings.postgres_password
POSTGRES_USER = settings.postgres_user

# Define the database URL
DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}/pgvector_db"
)

# Create an async engine and session
engine = create_async_engine(DATABASE_URL)
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
        print(pd.DataFrame(ratings_data).head(5))

    asyncio.run(main())
