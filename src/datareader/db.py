from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from core.config import settings

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
    """Async function to select data from DB

    Args:
        table_model (_type_): The target table to apply select.

    Returns:
        _type_: _description_
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(table_model))
        ratings = result.scalars().all()
        return ratings
