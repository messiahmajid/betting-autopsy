"""Scheduler for periodic data pulls."""

import asyncio
import argparse
from datetime import datetime

from src.models.database import Database
from src.ingest.polymarket import PolymarketClient
from src.ingest.metaculus import MetaculusClient
from src.ingest.manifold import ManifoldClient


async def run_full_ingest(
    include_polymarket: bool = True,
    include_metaculus: bool = True,
    include_manifold: bool = True,
    limit_per_platform: int = 500,
):
    """Run a full data ingest from all platforms."""
    db = Database()
    db.init_schema()

    print(f"Starting full ingest at {datetime.now()}")
    print("=" * 50)

    total_markets = 0

    if include_polymarket:
        print("\n--- Polymarket ---")
        client = PolymarketClient(db)
        try:
            count = await client.ingest_markets(
                limit=limit_per_platform, include_history=True, closed_only=True
            )
            total_markets += count
        except Exception as e:
            print(f"Polymarket error: {e}")
        finally:
            await client.close()

    if include_metaculus:
        print("\n--- Metaculus ---")
        client = MetaculusClient(db)
        try:
            count = await client.ingest_questions(
                limit=limit_per_platform, resolved_only=True, binary_only=True
            )
            total_markets += count
        except Exception as e:
            print(f"Metaculus error: {e}")
        finally:
            await client.close()

    if include_manifold:
        print("\n--- Manifold ---")
        client = ManifoldClient(db)
        try:
            count = await client.ingest_markets(limit=limit_per_platform, resolved_only=True)
            total_markets += count
        except Exception as e:
            print(f"Manifold error: {e}")
        finally:
            await client.close()

    print("\n" + "=" * 50)
    print(f"Ingest complete. Total markets: {total_markets}")

    # Print stats
    stats = db.get_stats()
    print("\nDatabase stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    db.close()
    return total_markets


async def run_periodic_ingest(interval_hours: float = 6):
    """Run ingestion periodically."""
    print(f"Starting periodic ingest (every {interval_hours} hours)")

    while True:
        try:
            await run_full_ingest()
        except Exception as e:
            print(f"Ingest error: {e}")

        print(f"\nSleeping for {interval_hours} hours...")
        await asyncio.sleep(interval_hours * 3600)


def main():
    parser = argparse.ArgumentParser(description="Data ingestion scheduler")
    parser.add_argument("--once", action="store_true", help="Run ingest once and exit")
    parser.add_argument(
        "--interval", type=float, default=6, help="Hours between ingests (default: 6)"
    )
    parser.add_argument(
        "--limit", type=int, default=500, help="Max markets per platform (default: 500)"
    )
    parser.add_argument("--polymarket", action="store_true", help="Only ingest Polymarket")
    parser.add_argument("--metaculus", action="store_true", help="Only ingest Metaculus")
    parser.add_argument("--manifold", action="store_true", help="Only ingest Manifold")
    args = parser.parse_args()

    # If specific platforms selected, only use those
    if args.polymarket or args.metaculus or args.manifold:
        include_poly = args.polymarket
        include_meta = args.metaculus
        include_mani = args.manifold
    else:
        include_poly = include_meta = include_mani = True

    if args.once:
        asyncio.run(
            run_full_ingest(
                include_polymarket=include_poly,
                include_metaculus=include_meta,
                include_manifold=include_mani,
                limit_per_platform=args.limit,
            )
        )
    else:
        asyncio.run(run_periodic_ingest(args.interval))


if __name__ == "__main__":
    main()
