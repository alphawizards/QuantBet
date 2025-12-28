"""
Async Database Connection Configuration.

Provides SQLAlchemy async engine and session factory.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool

from src.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Engine Configuration
# ============================================================================

def create_engine() -> AsyncEngine:
    """
    Create Async SQLAlchemy engine.

    Using NullPool for now to avoid issues if running in serverless/lambdas,
    but for a dedicated server, QueuePool (default) is better.
    Switching to default pool behavior for production readiness.
    """
    logger.info(f"Connecting to database at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")

    return create_async_engine(
        settings.ASYNC_DATABASE_URL,
        echo=settings.DEBUG,
        future=True,
        pool_pre_ping=True,
        # pool_size=20,  # Uncomment for QueuePool customization
        # max_overflow=10
    )


engine = create_engine()


# ============================================================================
# Session Factory
# ============================================================================

async_session_factory = async_sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get DB session.

    Yields:
        AsyncSession: Database session
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session rollback: {e}")
            raise
        finally:
            await session.close()
