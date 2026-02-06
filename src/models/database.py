"""DuckDB database connection and operations."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl

from src.models.schemas import Forecast, Market, MarketLink, Platform, Price

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "autopsy.duckdb"

SCHEMA_SQL = """
-- Markets/questions
CREATE TABLE IF NOT EXISTS markets (
    id VARCHAR PRIMARY KEY,
    platform VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    description TEXT,
    category VARCHAR,
    created_at TIMESTAMP,
    close_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution DOUBLE,
    resolution_source VARCHAR
);

-- Price/probability snapshots
CREATE TABLE IF NOT EXISTS prices (
    market_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    probability DOUBLE NOT NULL,
    volume DOUBLE,
    liquidity DOUBLE,
    source VARCHAR,
    PRIMARY KEY (market_id, timestamp)
);

-- Cross-platform market mappings
CREATE TABLE IF NOT EXISTS market_links (
    id VARCHAR PRIMARY KEY,
    event_description VARCHAR,
    polymarket_id VARCHAR,
    metaculus_id VARCHAR,
    manifold_id VARCHAR,
    predictit_id VARCHAR,
    kalshi_id VARCHAR
);

-- Forecaster predictions
CREATE TABLE IF NOT EXISTS forecasts (
    id VARCHAR PRIMARY KEY,
    market_id VARCHAR NOT NULL,
    forecaster_id VARCHAR,
    timestamp TIMESTAMP NOT NULL,
    prediction DOUBLE NOT NULL,
    is_community BOOLEAN DEFAULT FALSE
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_markets_platform ON markets(platform);
CREATE INDEX IF NOT EXISTS idx_markets_resolved ON markets(resolved_at);
CREATE INDEX IF NOT EXISTS idx_prices_market ON prices(market_id);
CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON prices(timestamp);
CREATE INDEX IF NOT EXISTS idx_forecasts_market ON forecasts(market_id);
"""


class Database:
    """DuckDB database interface for prediction market data."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def init_schema(self) -> None:
        """Create database tables."""
        self.conn.execute(SCHEMA_SQL)
        print(f"Database initialized at {self.db_path}")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Market operations ---

    def upsert_market(self, market: Market) -> None:
        """Insert or update a market."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO markets
            (id, platform, title, description, category, created_at, close_at,
             resolved_at, resolution, resolution_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                market.id,
                market.platform.value,
                market.title,
                market.description,
                market.category,
                market.created_at,
                market.close_at,
                market.resolved_at,
                market.resolution,
                market.resolution_source,
            ],
        )

    def upsert_markets(self, markets: list[Market]) -> None:
        """Bulk insert or update markets."""
        if not markets:
            return
        data = [
            (
                m.id,
                m.platform.value,
                m.title,
                m.description,
                m.category,
                m.created_at,
                m.close_at,
                m.resolved_at,
                m.resolution,
                m.resolution_source,
            )
            for m in markets
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO markets
            (id, platform, title, description, category, created_at, close_at,
             resolved_at, resolution, resolution_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )

    def get_market(self, market_id: str) -> Optional[Market]:
        """Get a market by ID."""
        result = self.conn.execute(
            "SELECT * FROM markets WHERE id = ?", [market_id]
        ).fetchone()
        if result is None:
            return None
        return self._row_to_market(result)

    def get_markets(
        self,
        platform: Optional[Platform] = None,
        resolved_only: bool = False,
        limit: Optional[int] = None,
    ) -> list[Market]:
        """Query markets with filters."""
        query = "SELECT * FROM markets WHERE 1=1"
        params = []

        if platform:
            query += " AND platform = ?"
            params.append(platform.value)

        if resolved_only:
            query += " AND resolution IS NOT NULL"

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        return [self._row_to_market(row) for row in results]

    def _row_to_market(self, row: tuple) -> Market:
        return Market(
            id=row[0],
            platform=Platform(row[1]),
            title=row[2],
            description=row[3],
            category=row[4],
            created_at=row[5],
            close_at=row[6],
            resolved_at=row[7],
            resolution=row[8],
            resolution_source=row[9],
        )

    # --- Price operations ---

    def insert_price(self, price: Price) -> None:
        """Insert a price snapshot (ignores duplicates)."""
        self.conn.execute(
            """
            INSERT OR IGNORE INTO prices
            (market_id, timestamp, probability, volume, liquidity, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                price.market_id,
                price.timestamp,
                price.probability,
                price.volume,
                price.liquidity,
                price.source,
            ],
        )

    def insert_prices(self, prices: list[Price]) -> None:
        """Bulk insert price snapshots."""
        if not prices:
            return
        data = [
            (p.market_id, p.timestamp, p.probability, p.volume, p.liquidity, p.source)
            for p in prices
        ]
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO prices
            (market_id, timestamp, probability, volume, liquidity, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            data,
        )

    def get_prices(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Get price history for a market as a Polars DataFrame."""
        query = "SELECT * FROM prices WHERE market_id = ?"
        params = [market_id]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"
        return self.conn.execute(query, params).pl()

    def get_latest_prices(self, platform: Optional[Platform] = None) -> pl.DataFrame:
        """Get the most recent price for each market."""
        query = """
            SELECT p.* FROM prices p
            INNER JOIN (
                SELECT market_id, MAX(timestamp) as max_ts
                FROM prices
                GROUP BY market_id
            ) latest ON p.market_id = latest.market_id AND p.timestamp = latest.max_ts
        """
        if platform:
            query = f"""
                SELECT p.* FROM prices p
                INNER JOIN markets m ON p.market_id = m.id
                INNER JOIN (
                    SELECT market_id, MAX(timestamp) as max_ts
                    FROM prices
                    GROUP BY market_id
                ) latest ON p.market_id = latest.market_id AND p.timestamp = latest.max_ts
                WHERE m.platform = '{platform.value}'
            """
        return self.conn.execute(query).pl()

    # --- Market link operations ---

    def upsert_market_link(self, link: MarketLink) -> None:
        """Insert or update a market link."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO market_links
            (id, event_description, polymarket_id, metaculus_id, manifold_id,
             predictit_id, kalshi_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                link.id,
                link.event_description,
                link.polymarket_id,
                link.metaculus_id,
                link.manifold_id,
                link.predictit_id,
                link.kalshi_id,
            ],
        )

    def get_market_links(self) -> list[MarketLink]:
        """Get all market links."""
        results = self.conn.execute("SELECT * FROM market_links").fetchall()
        return [
            MarketLink(
                id=row[0],
                event_description=row[1],
                polymarket_id=row[2],
                metaculus_id=row[3],
                manifold_id=row[4],
                predictit_id=row[5],
                kalshi_id=row[6],
            )
            for row in results
        ]

    # --- Forecast operations ---

    def insert_forecast(self, forecast: Forecast) -> None:
        """Insert a forecast."""
        self.conn.execute(
            """
            INSERT OR IGNORE INTO forecasts
            (id, market_id, forecaster_id, timestamp, prediction, is_community)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                forecast.id,
                forecast.market_id,
                forecast.forecaster_id,
                forecast.timestamp,
                forecast.prediction,
                forecast.is_community,
            ],
        )

    def insert_forecasts(self, forecasts: list[Forecast]) -> None:
        """Bulk insert forecasts."""
        if not forecasts:
            return
        data = [
            (f.id, f.market_id, f.forecaster_id, f.timestamp, f.prediction, f.is_community)
            for f in forecasts
        ]
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO forecasts
            (id, market_id, forecaster_id, timestamp, prediction, is_community)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            data,
        )

    def get_forecasts(self, market_id: str) -> pl.DataFrame:
        """Get forecasts for a market."""
        return self.conn.execute(
            "SELECT * FROM forecasts WHERE market_id = ? ORDER BY timestamp",
            [market_id],
        ).pl()

    # --- Analytics queries ---

    def get_resolved_markets_with_final_price(
        self, platform: Optional[Platform] = None
    ) -> pl.DataFrame:
        """Get resolved markets with their final probability before resolution."""
        query = """
            WITH final_prices AS (
                SELECT
                    p.market_id,
                    p.probability as final_probability,
                    ROW_NUMBER() OVER (PARTITION BY p.market_id ORDER BY p.timestamp DESC) as rn
                FROM prices p
                INNER JOIN markets m ON p.market_id = m.id
                WHERE m.resolution IS NOT NULL
                  AND p.timestamp <= COALESCE(m.resolved_at, m.close_at, p.timestamp)
            )
            SELECT
                m.id,
                m.platform,
                m.title,
                m.category,
                m.resolution,
                m.resolved_at,
                fp.final_probability
            FROM markets m
            INNER JOIN final_prices fp ON m.id = fp.market_id AND fp.rn = 1
            WHERE m.resolution IS NOT NULL
        """
        if platform:
            query += f" AND m.platform = '{platform.value}'"

        return self.conn.execute(query).pl()

    def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {}
        stats["total_markets"] = self.conn.execute(
            "SELECT COUNT(*) FROM markets"
        ).fetchone()[0]
        stats["resolved_markets"] = self.conn.execute(
            "SELECT COUNT(*) FROM markets WHERE resolution IS NOT NULL"
        ).fetchone()[0]
        stats["total_prices"] = self.conn.execute(
            "SELECT COUNT(*) FROM prices"
        ).fetchone()[0]
        stats["total_forecasts"] = self.conn.execute(
            "SELECT COUNT(*) FROM forecasts"
        ).fetchone()[0]
        stats["markets_by_platform"] = dict(
            self.conn.execute(
                "SELECT platform, COUNT(*) FROM markets GROUP BY platform"
            ).fetchall()
        )
        return stats


def main():
    parser = argparse.ArgumentParser(description="Database operations")
    parser.add_argument("--init", action="store_true", help="Initialize database schema")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    args = parser.parse_args()

    db = Database()

    if args.init:
        db.init_schema()
    elif args.stats:
        stats = db.get_stats()
        print("Database Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        parser.print_help()

    db.close()


if __name__ == "__main__":
    main()
