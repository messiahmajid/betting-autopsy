"""Manifold Markets API client."""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.models.schemas import Market, Platform, Price
from src.models.database import Database

MANIFOLD_API_URL = "https://api.manifold.markets/v0"


class ManifoldClient:
    """Client for fetching data from Manifold Markets."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_markets(
        self,
        limit: int = 100,
        before: Optional[str] = None,  # cursor for pagination
        sort: str = "created-time",
    ) -> list[dict]:
        """Fetch markets from Manifold API."""
        params = {"limit": limit, "sort": sort}
        if before:
            params["before"] = before

        response = await self.client.get(f"{MANIFOLD_API_URL}/markets", params=params)
        response.raise_for_status()
        return response.json()

    async def fetch_market(self, market_id: str) -> dict:
        """Fetch a single market by ID or slug."""
        response = await self.client.get(f"{MANIFOLD_API_URL}/market/{market_id}")
        response.raise_for_status()
        return response.json()

    async def fetch_market_positions(self, market_id: str) -> list[dict]:
        """Fetch positions/bets on a market."""
        response = await self.client.get(f"{MANIFOLD_API_URL}/market/{market_id}/positions")
        response.raise_for_status()
        return response.json()

    async def search_markets(
        self,
        term: str,
        limit: int = 100,
        filter_: str = "all",  # 'open', 'closed', 'resolved', 'all'
    ) -> list[dict]:
        """Search markets by term."""
        params = {"term": term, "limit": limit, "filter": filter_}
        response = await self.client.get(f"{MANIFOLD_API_URL}/search-markets", params=params)
        response.raise_for_status()
        return response.json()

    def _parse_market(self, data: dict) -> Optional[Market]:
        """Parse API response into Market model."""
        # Only handle binary markets
        outcome_type = data.get("outcomeType")
        if outcome_type != "BINARY":
            return None

        # Parse resolution
        resolution = None
        resolution_source = None
        if data.get("isResolved"):
            res = data.get("resolution")
            if res == "YES":
                resolution = 1.0
            elif res == "NO":
                resolution = 0.0
            elif res == "MKT":
                # Resolved to market probability
                resolution = data.get("resolutionProbability")
            resolution_source = "manifold"

        # Parse timestamps (Manifold uses milliseconds)
        created_at = None
        if data.get("createdTime"):
            created_at = datetime.fromtimestamp(data["createdTime"] / 1000, tz=timezone.utc)

        close_at = None
        if data.get("closeTime"):
            close_at = datetime.fromtimestamp(data["closeTime"] / 1000, tz=timezone.utc)

        resolved_at = None
        if data.get("resolutionTime"):
            resolved_at = datetime.fromtimestamp(
                data["resolutionTime"] / 1000, tz=timezone.utc
            )

        # Get category from groups
        category = None
        groups = data.get("groupSlugs", [])
        if groups:
            category = groups[0]

        return Market(
            id=f"manifold_{data['id']}",
            platform=Platform.MANIFOLD,
            title=data.get("question", ""),
            description=data.get("textDescription"),
            category=category,
            created_at=created_at,
            close_at=close_at,
            resolved_at=resolved_at,
            resolution=resolution,
            resolution_source=resolution_source,
        )

    def _parse_current_price(self, market_id: str, data: dict) -> Optional[Price]:
        """Parse current probability as a price point."""
        prob = data.get("probability")
        if prob is None:
            return None

        # Use last updated time or current time
        timestamp = datetime.now(tz=timezone.utc)
        if data.get("lastUpdatedTime"):
            timestamp = datetime.fromtimestamp(
                data["lastUpdatedTime"] / 1000, tz=timezone.utc
            )

        volume = data.get("volume")
        liquidity = data.get("totalLiquidity")

        return Price(
            market_id=market_id,
            timestamp=timestamp,
            probability=prob,
            volume=volume,
            liquidity=liquidity,
            source="manifold_api",
        )

    async def ingest_markets(
        self,
        limit: int = 500,
        resolved_only: bool = True,
    ) -> int:
        """Ingest markets from Manifold."""
        print(f"Fetching Manifold markets (limit={limit}, resolved_only={resolved_only})...")

        all_markets_data = []
        cursor = None
        batch_size = 100

        while len(all_markets_data) < limit:
            markets_data = await self.fetch_markets(
                limit=min(batch_size, limit - len(all_markets_data)),
                before=cursor,
            )
            if not markets_data:
                break

            if resolved_only:
                markets_data = [m for m in markets_data if m.get("isResolved")]

            all_markets_data.extend(markets_data)

            # Get cursor for next page
            if markets_data:
                cursor = markets_data[-1].get("id")
            else:
                break

            print(f"  Fetched {len(all_markets_data)} markets...")

        print(f"Processing {len(all_markets_data)} markets...")
        markets = []
        prices = []

        for data in all_markets_data:
            try:
                market = self._parse_market(data)
                if market:
                    markets.append(market)

                    # Get current price
                    price = self._parse_current_price(market.id, data)
                    if price:
                        prices.append(price)

            except Exception as e:
                print(f"  Error processing market {data.get('id')}: {e}")
                continue

        # Save to database
        self.db.upsert_markets(markets)
        self.db.insert_prices(prices)

        print(f"Saved {len(markets)} markets and {len(prices)} price points")
        return len(markets)


async def main():
    """Run ingestion."""
    db = Database()
    db.init_schema()

    client = ManifoldClient(db)
    try:
        await client.ingest_markets(limit=500, resolved_only=True)
    finally:
        await client.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
