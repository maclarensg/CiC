"""Smart routing example.

Demonstrates complexity-based model selection. Simple tasks use a fast
cheap model; complex reasoning tasks use a more capable one.

Run with:
    python examples/smart_routing.py
"""

import asyncio

from cic import CiCClient


async def main() -> None:
    # Different models for different workloads
    client = CiCClient(
        routing={
            "simple": "haiku",      # Fast, low-cost — greetings, lookups
            "moderate": "sonnet",   # Balanced — most tasks
            "complex": "opus",      # Heavy reasoning — analysis, long-form writing
        }
    )

    messages_simple = [{"role": "user", "content": "What is 2 + 2?"}]
    messages_complex = [
        {
            "role": "user",
            "content": (
                "Analyse the trade-offs between event-driven and request-response "
                "architecture patterns for a high-throughput financial trading system."
            ),
        }
    ]

    # Simple task → haiku
    client.set_complexity("simple")
    print(f"Simple task using: {client.active_model}")
    result = await client.achat(messages_simple)
    print(f"Answer: {result.content}\n")

    # Complex task → opus
    client.set_complexity("complex")
    print(f"Complex task using: {client.active_model}")
    result = await client.achat(messages_complex)
    print(f"Answer (first 200 chars): {(result.content or '')[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
