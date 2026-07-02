"""Context compressor: summarize distant blocks to fit, never silently truncate."""

import pytest

from config import load_config
from context.compressor import Block, CompressionError, Compressor, P_CRITICAL, P_DISTANT_SRC


def _cfg(budget: int):
    c = load_config()
    c.budgets = {"planner": budget}
    c.reserve_for_output = 0
    return c


def test_no_compression_when_under_budget(bus):
    comp = Compressor(_cfg(100000), bus)
    out = comp.fit("planner", [Block("A", "short", P_CRITICAL)])
    assert "short" in out
    assert not bus.of_type("compression")


def test_compresses_distant_block_and_logs_event(bus):
    comp = Compressor(_cfg(1100), bus)  # usable ~1100, headroom ~1045 tokens
    big = "x " * 4000
    blocks = [
        Block("STEERING", "keep me", P_CRITICAL),
        Block("FILE far", big, P_DISTANT_SRC, summary="one-line summary"),
    ]
    out = comp.fit("planner", blocks)
    assert "one-line summary" in out  # the distant block was summarized
    assert "keep me" in out  # the critical block stayed verbatim
    assert bus.of_type("compression")  # a decision event was emitted


def test_raises_when_even_full_compression_wont_fit(bus):
    comp = Compressor(_cfg(1100), bus)
    big = "x " * 4000
    with pytest.raises(CompressionError):
        comp.fit("planner", [Block("CRIT", big, P_CRITICAL)])  # critical + no summary
    assert bus.of_type("compression_failure")
