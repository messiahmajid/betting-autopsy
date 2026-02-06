"""Metaculus API client for forecasting data."""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import uuid4

import httpx

from src.models.schemas import Forecast, Market, Platform, Price
from src.models.database import Database

METACULUS_API_URL = "https://www.metaculus.com/api2"


class MetaculusClient:
    """Client for fetching data from Metaculus."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": "BettingAutopsy/1.0"},
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_questions(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "-publish_time",
    ) -> dict:
        """Fetch questions from Metaculus API."""
        params = {
            "limit": limit,
            "offset": offset,
            "order_by": order_by,
        }

        response = await self.client.get(f"{METACULUS_API_URL}/questions/", params=params)
        response.raise_for_status()
        return response.json()

    async def fetch_question(self, question_id: int) -> dict:
        """Fetch a single question by ID."""
        response = await self.client.get(f"{METACULUS_API_URL}/questions/{question_id}/")
        response.raise_for_status()
        return response.json()

    async def fetch_question_predictions(self, question_id: int) -> dict:
        """Fetch prediction history for a question."""
        response = await self.client.get(
            f"{METACULUS_API_URL}/questions/{question_id}/predictions/"
        )
        response.raise_for_status()
        return response.json()

    def _parse_question(self, data: dict) -> Optional[Market]:
        """Parse API response into Market model."""
        # Get nested question data
        question_data = data.get("question", {})

        # Only handle binary questions
        question_type = question_data.get("type", data.get("type"))
        if question_type not in ("binary", "forecast", None):
            return None

        # Parse resolution
        resolution = None
        resolution_source = None
        res_value = question_data.get("resolution", data.get("resolution"))
        if res_value is not None:
            if res_value == "yes":
                resolution = 1.0
            elif res_value == "no":
                resolution = 0.0
            elif res_value == "ambiguous":
                return None  # Skip ambiguous resolutions
            else:
                try:
                    resolution = float(res_value)
                    resolution_source = "metaculus"
                except (ValueError, TypeError):
                    pass
            resolution_source = "metaculus"

        # Parse timestamps
        created_at = None
        publish_time = data.get("published_at") or data.get("publish_time")
        if publish_time:
            try:
                created_at = datetime.fromisoformat(
                    publish_time.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        close_at = None
        close_time = data.get("actual_close_time") or data.get("scheduled_close_time")
        if close_time:
            try:
                close_at = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            except ValueError:
                pass

        resolved_at = None
        resolve_time = data.get("actual_resolve_time") or question_data.get("actual_resolve_time")
        if resolve_time:
            try:
                resolved_at = datetime.fromisoformat(
                    resolve_time.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Get category from projects
        category = None
        projects = data.get("projects", {})
        if isinstance(projects, dict):
            categories = projects.get("category", [])
            if categories and isinstance(categories, list) and len(categories) > 0:
                category = categories[0].get("name")

        return Market(
            id=f"metaculus_{data['id']}",
            platform=Platform.METACULUS,
            title=data.get("title", ""),
            description=data.get("description"),
            category=category,
            created_at=created_at,
            close_at=close_at,
            resolved_at=resolved_at,
            resolution=resolution,
            resolution_source=resolution_source,
        )

    def _parse_community_prediction(self, market_id: str, data: dict) -> Optional[Price]:
        """Parse community prediction as a price point."""
        # Try new API format first
        question_data = data.get("question", {})
        aggregations = question_data.get("aggregations", {})
        unweighted = aggregations.get("unweighted", {})
        latest = unweighted.get("latest", {})

        prob = None

        # Get the center value (median prediction)
        centers = latest.get("centers", [])
        if centers and len(centers) > 0:
            prob = centers[0]

        # Fallback to old format
        if prob is None:
            prediction = data.get("community_prediction")
            if prediction is not None:
                if isinstance(prediction, dict):
                    prob = prediction.get("full", {}).get("q2")
                elif isinstance(prediction, (int, float)):
                    prob = float(prediction)

        if prob is None:
            return None

        # Use resolution time or current time for timestamp
        timestamp = datetime.now()
        resolve_time = data.get("actual_resolve_time") or question_data.get("actual_resolve_time")
        if resolve_time:
            try:
                timestamp = datetime.fromisoformat(resolve_time.replace("Z", "+00:00"))
            except ValueError:
                pass

        return Price(
            market_id=market_id,
            timestamp=timestamp,
            probability=prob,
            source="metaculus_community",
        )

    def _parse_forecast(
        self, market_id: str, prediction_data: dict, forecaster_id: Optional[str] = None
    ) -> Optional[Forecast]:
        """Parse a prediction into Forecast model."""
        try:
            timestamp = datetime.fromisoformat(
                prediction_data["time"].replace("Z", "+00:00")
            )
            prediction = float(prediction_data["x"])

            return Forecast(
                id=str(uuid4()),
                market_id=market_id,
                forecaster_id=forecaster_id,
                timestamp=timestamp,
                prediction=prediction,
                is_community=forecaster_id is None,
            )
        except (KeyError, ValueError):
            return None

    async def ingest_questions(
        self,
        limit: int = 500,
        resolved_only: bool = True,
        binary_only: bool = True,
    ) -> int:
        """Ingest questions from Metaculus."""
        print(f"Fetching Metaculus questions (limit={limit}, resolved_only={resolved_only})...")

        all_questions = []
        offset = 0
        batch_size = 100
        max_fetches = 20  # Safety limit

        fetches = 0
        while len(all_questions) < limit and fetches < max_fetches:
            fetches += 1
            response_data = await self.fetch_questions(
                limit=batch_size,
                offset=offset,
            )

            questions = response_data.get("results", [])
            if not questions:
                break

            # Filter client-side for resolved questions
            for q in questions:
                if resolved_only and not q.get("resolved"):
                    continue
                if binary_only and q.get("question", {}).get("type") not in (None, "binary"):
                    # Skip non-binary if we can detect them
                    pass
                all_questions.append(q)

            offset += batch_size
            print(f"  Fetched {offset} questions, {len(all_questions)} resolved...")

            if not response_data.get("next"):
                break

            if len(all_questions) >= limit:
                break

        print(f"Processing {len(all_questions)} questions...")
        markets = []
        prices = []

        for data in all_questions:
            try:
                market = self._parse_question(data)
                if market:
                    markets.append(market)

                    # Get community prediction as price point
                    price = self._parse_community_prediction(market.id, data)
                    if price:
                        prices.append(price)

            except Exception as e:
                print(f"  Error processing question {data.get('id')}: {e}")
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

    client = MetaculusClient(db)
    try:
        await client.ingest_questions(limit=500, resolved_only=True, binary_only=True)
    finally:
        await client.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
