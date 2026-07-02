"""Ephemeral-conversation packing + oversized tool-output capping."""

from context.conversation import cap_tool_output, pack_conversation


def test_pack_keeps_system_and_newest():
    msgs = [{"role": "system", "content": "SYS"}]
    msgs += [{"role": "user", "content": f"m{i} " + "word " * 50} for i in range(20)]
    packed = pack_conversation(msgs, 200)
    assert packed[0]["role"] == "system" and packed[0]["content"] == "SYS"
    assert packed[-1]["content"].startswith("m19")  # newest retained
    assert len(packed) < len(msgs)  # older rounds dropped to fit budget


def test_pack_keeps_newest_even_if_alone_too_big():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "word " * 5000}]
    packed = pack_conversation(msgs, 100)
    assert packed[0]["role"] == "system"
    assert packed[-1]["content"].startswith("word")  # degrades gracefully, doesn't vanish


def test_cap_tool_output_collapses_big_with_marker():
    out = cap_tool_output("A" * 60000, max_tokens=500, head_tail_tokens=100)
    assert len(out) < 60000 and "OMITTED" in out


def test_cap_tool_output_passthrough_small():
    assert cap_tool_output("tiny") == "tiny"
