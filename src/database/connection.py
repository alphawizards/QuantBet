"""
Database Connection Configuration with Connection Pooling.

This module provides SQLAlchemy engine configuration with proper connection
pooling for production workloads.

Features:
    - Connection pooling with configurable size
    - Connection recycling for long-running applications
    - Pre-ping to detect stale connections
    - Overflow handling for burst traffic
"""

import os
import logging
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_POOL_SIZE = 5        # Number of connections to keep in pool
DEFAULT_MAX_OVERFLOW = 10    # Additional connections allowed during burst
DEFAULT_POOL_TIMEOUT = 30    # Seconds to wait for a connection
DEFAULT_POOL_RECYCLE = 1800  # Recycle connections after 30 minutes
DEFAULT_POOL_PRE_PING = True # Check connection validity before use


def get_database_url() -> str:
    """
    Get database URL from environment variables.
    
    Returns:
        PostgreSQL connection string
        
    Raises:
        ValueError: If DATABASE_URL is not set
    """
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        # Try to construct from individual components
        user = os.getenv("POSTGRES_USER", "quantbet")
        password = os.getenv("POSTGRES_PASSWORD", "")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "quantbet")
        
        if not password:
            raise ValueError(
                "DATABASE_URL or POSTGRES_PASSWORD must be set in environment"
            )
        
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    return database_url


def create_engine_with_pool(
    database_url: Optional[str] = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_overflow: int = DEFAULT_MAX_OVERFLOW,
    pool_timeout: int = DEFAULT_POOL_TIMEOUT,
    pool_recycle: int = DEFAULT_POOL_RECYCLE,
    pool_pre_ping: bool = DEFAULT_POOL_PRE_PING,
    echo: bool = False
) -> Engine:
    """
    Create SQLAlchemy engine with connection pooling.
    
    Args:
        database_url: PostgreSQL connection string. If None, reads from env.
        pool_size: Number of connections to maintain in the pool.
        max_overflow: Maximum number of connections to create beyond pool_size.
        pool_timeout: Seconds to wait for a connection from the pool.
        pool_recycle: Seconds after which to recycle connections.
        pool_pre_ping: Whether to test connections before using them.
        echo: Whether to log SQL statements (for debugging).
    
    Returns:
        Configured SQLAlchemy Engine with QueuePool
        
    Example:
        >>> engine = create_engine_with_pool(pool_size=10, max_overflow=20)
        >>> with engine.connect() as conn:
        ...     result = conn.execute(text("SELECT 1"))
    """
    if database_url is None:
        database_url = get_database_url()
    
    logger.info(
        f"Creating database engine with pool_size={pool_size}, "
        f"max_overflow={max_overflow}, pool_recycle={pool_recycle}s"
    )
    
    engine = create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        pool_pre_ping=pool_pre_ping,
        echo=echo,
        # Additional performance settings
        connect_args={
            "options": "-c timezone=utc",
            "application_name": "quantbet_api",
        }
    )
    
    # Register event listeners for monitoring
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_connection, connection_record):
        logger.debug("New database connection established")
    
    @event.listens_for(engine, "checkout")
    def on_checkout(dbapi_connection, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")
    
    @event.listens_for(engine, "checkin")
    def on_checkin(dbapi_connection, connection_record):
        logger.debug("Connection returned to pool")
    
    return engine


# ============================================================================
# Session Factory
# ============================================================================

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Get or create the database engine singleton."""
    global _engine
    if _engine is None:
        _engine = create_engine_with_pool()
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory singleton."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    return _session_factory


def get_session() -> Session:
    """
    Create a new database session.
    
    Returns:
        SQLAlchemy Session object
        
    Usage:
        >>> session = get_session()
        >>> try:
        ...     # Use session
        ...     session.commit()
        ... finally:
        ...     session.close()
    """
    factory = get_session_factory()
    return factory()


class DatabaseSession:
    """
    Context manager for database sessions.
    
    Automatically handles commit/rollback and cleanup.
    
    Usage:
        >>> with DatabaseSession() as session:
        ...     result = session.query(Team).all()
    """
    
    def __init__(self):
        self.session: Optional[Session] = None
    
    def __enter__(self) -> Session:
        self.session = get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            try:
                if exc_type is None:
                    self.session.commit()
                else:
                    self.session.rollback()
                    logger.error(f"Session rolled back due to: {exc_val}")
            finally:
                self.session.close()
        return False  # Don't suppress exceptions


# ============================================================================
# Pool Status and Monitoring
# ============================================================================

def get_pool_status() -> dict:
    """
    Get current connection pool status.
    
    Returns:
        Dictionary with pool statistics
    """
    engine = get_engine()
    pool = engine.pool
    
    return {
        "pool_size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "checked_in": pool.checkedin(),
        "invalidated": pool.invalidatedcount() if hasattr(pool, 'invalidatedcount') else 0,
    }


def dispose_engine():
    """
    Dispose of all pooled connections.
    
    Call this during application shutdown or when reconfiguring.
    """
    global _engine, _session_factory
    
    if _engine:
        logger.info("Disposing database engine and connection pool")
        _engine.dispose()
        _engine = None
        _session_factory = None
