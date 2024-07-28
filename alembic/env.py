from logging.config import fileConfig
import pgvector
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from table_models.ml_10m.settings import Base

from alembic import context
import sqlalchemy
from sqlalchemy.orm import Session


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

        create_vector_extension(connection)


def create_vector_extension(connection) -> None:
    try:
        with Session(connection) as session:  # type: ignore[arg-type]
            # The advisor lock fixes issue arising from concurrent
            # creation of the vector extension.
            # https://github.com/langchain-ai/langchain/issues/12933
            # For more information see:
            # https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS
            statement = sqlalchemy.text(
                "BEGIN;" "CREATE EXTENSION IF NOT EXISTS vector;" "COMMIT;"
            )
            session.execute(statement)
            session.commit()
    except Exception as e:
        raise Exception(f"Failed to create vector extension: {e}") from e


def do_run_migrations(connection) -> None:
    # Need to hack the "vector" type into postgres dialect schema types.
    # Otherwise, `alembic check` does not recognize the type

    # This line of code registers the custom vector type with SQLAlchemy so that SQLAlchemy knows how to
    # handle this custom type when interacting with the database schema.
    # It tells SQLAlchemy that the vector type in PostgreSQL should be mapped
    # to the pgvector.sqlalchemy.Vector type in Python.
    # ref: https://github.com/sqlalchemy/alembic/discussions/1324
    connection.dialect.ischema_names["vector"] = pgvector.sqlalchemy.Vector

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
