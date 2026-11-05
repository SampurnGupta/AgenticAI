"""Task monitoring and logging callbacks for the Tangent Investment Committee."""

import json
import time
from typing import Any, Optional

from common.logging import get_logger
from common.redis_client import TangentRedisClient, get_redis_client

logger = get_logger("tangent-committee-monitor")


class CommitteeMonitorCallback:
    """Tracks step latencies, reasoning, tool executions, and streams telemetry to Redis."""

    def __init__(self, run_id: str, redis_client: Optional[TangentRedisClient] = None):
        self.run_id = run_id
        self.redis = redis_client or get_redis_client()
        self.step_history: list[dict[str, Any]] = []
        self._start_time = time.perf_counter()

    async def log_step(
        self,
        agent_id: str,
        step_name: str,
        duration_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        tool_call: Optional[dict[str, Any]] = None,
        reasoning: Optional[str] = None,
    ) -> None:
        """Record step metadata and stream to Redis."""
        step_data = {
            "run_id": self.run_id,
            "agent_id": agent_id,
            "step_name": step_name,
            "duration_ms": round(duration_ms, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 6),
            "tool_call": tool_call,
            "reasoning": reasoning[:200] if reasoning else None,
            "timestamp": time.time(),
        }
        self.step_history.append(step_data)
        logger.info("Committee Step [%s] %s finished in %.1fms", agent_id, step_name, duration_ms)

        # Stream to Redis if available
        try:
            client = await self.redis._get_client()
            if client:
                telemetry_key = f"tangent:telemetry:committee:{self.run_id}"
                await client.rpush(telemetry_key, json.dumps(step_data))
                await client.expire(telemetry_key, 86400)  # 24h retention
                # Also increment global telemetry counter
                await client.incr("tangent:telemetry:total_steps")
        except Exception as ex:
            logger.debug("Redis telemetry push skipped: %s", ex)

    async def record_observatory_call(
        self,
        model: str,
        persona: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        """Stream model call to the LLM Observatory leaderboard."""
        try:
            await self.redis.record_llm_metric(
                model=model,
                persona=persona,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
            )
        except Exception as ex:
            logger.debug("Observatory metric recording error: %s", ex)
