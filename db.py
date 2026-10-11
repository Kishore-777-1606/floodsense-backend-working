"""
FloodSense MongoDB Persistence Layer
Manages database connection, collection handles, and indexing.
Supports both local MongoDB and MongoDB Atlas via MONGODB_URI.
"""

import os
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("floodsense.db")
logging.basicConfig(level=logging.INFO)

DEFAULT_URI = "mongodb://localhost:27017/floodsense"
DEFAULT_DB_NAME = "floodsense"

_client = None
_db = None


def get_client() -> MongoClient:
    """Returns singleton MongoClient instance with safe timeout configuration."""
    global _client
    if _client is None:
        mongo_uri = os.getenv("MONGODB_URI", DEFAULT_URI)
        # 3-second timeout for server selection to prevent hanging on connection failure
        _client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=30000,
            appname="FloodSenseRiskEngine",
        )
    return _client


def get_db():
    """Returns the FloodSense database instance."""
    global _db
    if _db is None:
        client = get_client()
        db_name = os.getenv("MONGODB_DB_NAME", DEFAULT_DB_NAME)
        # If the URI specifies a database path, extract or default
        _db = client.get_default_database() if client.get_default_database() is not None else client[db_name]
    return _db


def is_connected() -> bool:
    """Verifies that the MongoDB server is reachable."""
    try:
        client = get_client()
        client.admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
        logger.warning(f"MongoDB connection check failed: {e}")
        return False


def init_db():
    """
    Initializes collections and creates performance-critical indexes.
    - rainfall_observations: compound index on (ward_no, timestamp)
    - rainfall_daily_summary: compound unique index on (ward_no, date)
    """
    if not is_connected():
        logger.error("Cannot initialize DB: MongoDB server unreachable.")
        return False

    db = get_db()

    # 1. Raw Observations Collection
    obs_col = db["rainfall_observations"]
    obs_col.create_index([("ward_no", ASCENDING), ("timestamp", DESCENDING)], name="idx_ward_timestamp")
    obs_col.create_index([("timestamp", DESCENDING)], name="idx_timestamp_desc")

    # 2. Daily Summary Collection (Longer-term memory aggregation)
    daily_col = db["rainfall_daily_summary"]
    daily_col.create_index([("ward_no", ASCENDING), ("date", DESCENDING)], unique=True, name="idx_ward_date_unique")

    logger.info("✅ MongoDB collections and indexes initialized successfully.")
    return True


def close_connection():
    """Closes client connection cleanly."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
