"""Thinking-mode resolution: auto-detect, coding/planning toggle, precedence."""

from config import (
    THINK_DEEPSEEK,
    THINK_NONE,
    THINK_QWEN3_OFF,
    THINK_QWEN3_ON,
    _toggle_thinking,
    detect_thinking_mode,
    load_config,
)


def test_auto_detect_from_model_name():
    assert detect_thinking_mode("ollama/qwen3-coder:30b") == THINK_QWEN3_ON
    assert detect_thinking_mode("ollama/qwen3.6:27b") == THINK_QWEN3_ON
    assert detect_thinking_mode("ollama/qwen2.5-coder:14b") == THINK_NONE
    assert detect_thinking_mode("ollama/llama3.2:3b") == THINK_NONE
    assert detect_thinking_mode("ollama/deepseek-r1:32b") == THINK_DEEPSEEK


def test_toggle_only_flips_supported_families():
    assert _toggle_thinking("ollama/qwen3-coder:30b", True) == THINK_QWEN3_ON
    assert _toggle_thinking("ollama/qwen3-coder:30b", False) == THINK_QWEN3_OFF
    assert _toggle_thinking("ollama/llama3.2:3b", True) == THINK_NONE  # no thinking to enable
    assert _toggle_thinking("ollama/deepseek-r1:32b", False) == THINK_NONE


def test_precedence_explicit_over_toggle_over_auto():
    cfg = load_config()
    cfg.models = {
        "implementer": "ollama/qwen3-coder:30b",
        "planner": "ollama/qwen3.6:27b",
        "intake": "ollama/llama3.2:3b",
    }
    # shipped config.yaml has coding/planning = true -> the coder thinks
    cfg.think_coding = True
    assert cfg.thinking_mode_for("implementer") == THINK_QWEN3_ON
    # toggle off -> /no_think
    cfg.think_coding = False
    assert cfg.thinking_mode_for("implementer") == THINK_QWEN3_OFF
    # an explicit model_options entry wins over the toggle
    cfg.thinking = {"implementer": THINK_QWEN3_ON}
    assert cfg.thinking_mode_for("implementer") == THINK_QWEN3_ON
    # a phase in neither group falls back to auto-detect
    assert cfg.thinking_mode_for("intake") == THINK_NONE
