"""Polymarket API client using their CLOB API and Gamma subgraph."""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.models.schemas import Market, Platform, Price
from src.models.database import Database

# Polymarket CLOB API (Central Limit Order Book)
CLOB_API_URL = "https://clob.polymarket.com"

# Gamma API for market data
GAMMA_API_URL = "https://gamma-api.polymarket.com"


class PolymarketClient:
    """Client for fetching data from Polymarket."""

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
        offset: int = 0,
        closed: Optional[bool] = None,
        active: Optional[bool] = None,
    ) -> list[dict]:
        """Fetch markets from Gamma API."""
        params = {
            "limit": limit,
            "offset": offset,
        }
        if closed is not None:
            params["closed"] = str(closed).lower()
        if active is not None:
            params["active"] = str(active).lower()

        response = await self.client.get(f"{GAMMA_API_URL}/markets", params=params)
        response.raise_for_status()
        return response.json()

    async def fetch_market(self, condition_id: str) -> dict:
        """Fetch a single market by condition ID."""
        response = await self.client.get(f"{GAMMA_API_URL}/markets/{condition_id}")
        response.raise_for_status()
        return response.json()

    async def fetch_events(
        self,
        limit: int = 100,
        offset: int = 0,
        closed: Optional[bool] = None,
    ) -> list[dict]:
        """Fetch events (which may contain multiple markets)."""
        params = {"limit": limit, "offset": offset}
        if closed is not None:
            params["closed"] = str(closed).lower()

        response = await self.client.get(f"{GAMMA_API_URL}/events", params=params)
        response.raise_for_status()
        return response.json()

    async def fetch_prices(self, token_id: str) -> dict:
        """Fetch current price/order book for a token."""
        response = await self.client.get(f"{CLOB_API_URL}/price", params={"token_id": token_id})
        response.raise_for_status()
        return response.json()

    async def fetch_price_history(
        self,
        clob_token_id: str,
        fidelity: int = 60,  # minutes between data points
    ) -> list[dict]:
        """Fetch historical prices for a token."""
        params = {"market": clob_token_id, "interval": "all", "fidelity": fidelity}
        response = await self.client.get(f"{CLOB_API_URL}/prices-history", params=params)
        response.raise_for_status()
        return response.json().get("history", [])

    def _parse_market(self, data: dict) -> Market:
        """Parse API response into Market model."""
        # Handle resolution
        resolution = None
        resolution_source = None

        if data.get("closed"):
            # Determine resolution from outcome prices
            outcome_prices = data.get("outcomePrices", "")
            if outcome_prices:
                try:
                    # Parse prices like '["0.99", "0.01"]' or '[0.99, 0.01]'
                    prices_str = outcome_prices.strip("[]")
                    prices = [float(p.strip().strip('"')) for p in prices_str.split(",") if p.strip()]

                    if len(prices) >= 1:
                        yes_price = prices[0]
                        # If YES price is near 1, YES won; if near 0, NO won
                        # Treat anything > 0.5 as YES resolution
                        if yes_price > 0.5:
                            resolution = 1.0
                        elif yes_price < 0.5:
                            resolution = 0.0
                        else:
                            # Exactly 0.5 or ambiguous - use raw value
                            resolution = yes_price
                        resolution_source = data.get("resolutionSource") or "polymarket_outcome_price"
                except (ValueError, IndexError):
                    pass

        # Parse timestamps
        created_at = None
        if data.get("createdAt"):
            created_at = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))

        close_at = None
        if data.get("endDate"):
            close_at = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))

        resolved_at = None
        if data.get("closed") and close_at:
            resolved_at = close_at

        return Market(
            id=f"polymarket_{data['conditionId']}",
            platform=Platform.POLYMARKET,
            title=data.get("question", data.get("title", "")),
            description=data.get("description"),
            category=data.get("category"),
            created_at=created_at,
            close_at=close_at,
            resolved_at=resolved_at,
            resolution=resolution,
            resolution_source=resolution_source,
        )

    def _parse_prices(self, market_id: str, history: list[dict]) -> list[Price]:
        """Parse price history into Price models."""
        prices = []
        for point in history:
            try:
                timestamp = datetime.fromtimestamp(point["t"], tz=timezone.utc)
                probability = float(point["p"])
                prices.append(
                    Price(
                        market_id=market_id,
                        timestamp=timestamp,
                        probability=probability,
                        source="polymarket_history",
                    )
                )
            except (KeyError, ValueError):
                continue
        return prices

    def _parse_final_price(self, market_id: str, data: dict) -> Optional[Price]:
        """Extract final price from market data for closed markets."""
        if not data.get("closed"):
            return None

        outcome_prices = data.get("outcomePrices", "")
        if not outcome_prices:
            return None

        try:
            prices_str = outcome_prices.strip("[]")
            prices = [float(p.strip().strip('"')) for p in prices_str.split(",") if p.strip()]

            if not prices:
                return None

            # YES outcome price is the probability
            yes_price = prices[0]

            # Get timestamp from endDate
            timestamp = datetime.now(timezone.utc)
            if data.get("endDate"):
                try:
                    timestamp = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
                except ValueError:
                    pass

            return Price(
                market_id=market_id,
                timestamp=timestamp,
                probability=yes_price,
                source="polymarket_final",
            )
        except (ValueError, IndexError):
            return None

    async def ingest_markets(
        self,
        limit: int = 500,
        include_history: bool = True,
        closed_only: bool = False,
    ) -> int:
        """Ingest markets and optionally their price history."""
        print(f"Fetching Polymarket markets (limit={limit}, closed_only={closed_only})...")

        all_markets = []
        offset = 0
        batch_size = 100

        while len(all_markets) < limit:
            markets_data = await self.fetch_markets(
                limit=min(batch_size, limit - len(all_markets)),
                offset=offset,
                closed=True if closed_only else None,
            )
            if not markets_data:
                break
            all_markets.extend(markets_data)
            offset += batch_size
            print(f"  Fetched {len(all_markets)} markets...")

        print(f"Processing {len(all_markets)} markets...")
        markets = []
        all_prices = []

        for data in all_markets:
            try:
                market = self._parse_market(data)
                markets.append(market)

                prices_added = False
                # Fetch price history if requested and token ID available
                if include_history:
                    clob_token_ids = data.get("clobTokenIds", "")
                    if clob_token_ids:
                        token_ids = [
                            t.strip() for t in clob_token_ids.strip("[]").split(",") if t.strip()
                        ]
                        if token_ids:
                            try:
                                history = await self.fetch_price_history(token_ids[0])
                                prices = self._parse_prices(market.id, history)
                                if prices:
                                    all_prices.extend(prices)
                                    prices_added = True
                            except httpx.HTTPError:
                                pass  # Skip if history not available

                # Always add the final price for closed markets if we don't have history
                if not prices_added:
                    final_price = self._parse_final_price(market.id, data)
                    if final_price:
                        all_prices.append(final_price)

            except Exception as e:
                print(f"  Error processing market: {e}")
                continue

        # Save to database
        self.db.upsert_markets(markets)
        self.db.insert_prices(all_prices)

        print(f"Saved {len(markets)} markets and {len(all_prices)} price points")
        return len(markets)


async def main():
    """Run ingestion."""
    db = Database()
    db.init_schema()

    client = PolymarketClient(db)
    try:
        await client.ingest_markets(limit=200, include_history=True, closed_only=True)
    finally:
        await client.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
