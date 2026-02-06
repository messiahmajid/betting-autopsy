"""Pydantic models for prediction market data."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Platform(str, Enum):
    """Supported prediction market platforms."""

    POLYMARKET = "polymarket"
    METACULUS = "metaculus"
    MANIFOLD = "manifold"
    PREDICTIT = "predictit"
    KALSHI = "kalshi"


class Market(BaseModel):
    """A prediction market or forecasting question."""

    id: str
    platform: Platform
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    close_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="1.0 = yes, 0.0 = no, None = unresolved"
    )
    resolution_source: Optional[str] = None

    @property
    def is_resolved(self) -> bool:
        return self.resolution is not None


class Price(BaseModel):
    """A price/probability snapshot for a market."""

    market_id: str
    timestamp: datetime
    probability: float = Field(ge=0.0, le=1.0)
    volume: Optional[float] = Field(default=None, ge=0.0)
    liquidity: Optional[float] = Field(default=None, ge=0.0)
    source: str = "api"


class MarketLink(BaseModel):
    """Links the same event across multiple platforms."""

    id: str
    event_description: str
    polymarket_id: Optional[str] = None
    metaculus_id: Optional[str] = None
    manifold_id: Optional[str] = None
    predictit_id: Optional[str] = None
    kalshi_id: Optional[str] = None

    def get_platform_ids(self) -> dict[Platform, str]:
        """Return dict of platform -> market_id for linked markets."""
        result = {}
        if self.polymarket_id:
            result[Platform.POLYMARKET] = self.polymarket_id
        if self.metaculus_id:
            result[Platform.METACULUS] = self.metaculus_id
        if self.manifold_id:
            result[Platform.MANIFOLD] = self.manifold_id
        if self.predictit_id:
            result[Platform.PREDICTIT] = self.predictit_id
        if self.kalshi_id:
            result[Platform.KALSHI] = self.kalshi_id
        return result


class Forecast(BaseModel):
    """An individual forecaster's prediction."""

    id: str
    market_id: str
    forecaster_id: Optional[str] = None
    timestamp: datetime
    prediction: float = Field(ge=0.0, le=1.0)
    is_community: bool = False


class CalibrationBucket(BaseModel):
    """A bucket in a calibration curve."""

    bin_start: float
    bin_end: float
    predicted_mean: float
    actual_frequency: float
    count: int
    std_error: Optional[float] = None


class ArbitrageOpportunity(BaseModel):
    """A detected arbitrage opportunity."""

    market_link_id: str
    timestamp: datetime
    platform_a: Platform
    platform_b: Platform
    prob_a: float
    prob_b: float
    profit_pct: float
    stake_a: float
    stake_b: float
    arb_type: str  # 'yes_a_no_b' or 'no_a_yes_b'


class PortfolioPosition(BaseModel):
    """A position in a betting portfolio."""

    market_id: str
    platform: Platform
    direction: str  # 'yes' or 'no'
    stake: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
