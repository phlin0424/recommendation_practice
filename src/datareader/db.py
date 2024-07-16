from core.config import settings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

DB_HOST = "localhost"
# DB_HOST = settings.db_host
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
async def fetch_datas(table_model):
    """Async function to select *ALL* the data from DB

    Args:
        table_model (_type_): The target table to apply select.

    Returns:
        _type_: _description_
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(table_model))
        ratings = result.scalars().all()
        return ratings


# Async function to run the sql to fetch the data from the database
async def get_tables(sql_query: str, args: dict[int, any]):
    """Async function to query data from DB based on SQL query

    Args:
        sql_query (str): _description_
        args (dict[int, any]): _description_

    Returns:
        _type_: _description_
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(text(sql_query), args)
        return result.fetchall()


if __name__ == "__main__":
    import asyncio

    with open("../sql/integrated_tables.sql", "r") as f:
        sql_query = f.read()

    async def main():
        args = {"user_num": 10}
        tables = await get_tables(sql_query=sql_query, args=args)
        for row in tables:
            print(row.movie_title)

    asyncio.run(main())
