"""Tolerant tool-call parser — the riskiest heuristic in the tool."""

from llm.tool_parser import extract_all_tool_calls, extract_json, extract_tool_call


def test_clean_tool_call():
    txt = '<agentic_call>{"tool":"write_file","args":{"path":"a.py","content":"print(1)","summary":"s"}}</agentic_call>'
    calls = extract_all_tool_calls(txt)
    assert len(calls) == 1
    c = calls[0]
    assert c.name == "write_file" and c.args["path"] == "a.py" and c.is_known


def test_multiple_calls_in_document_order():
    txt = (
        '<agentic_call>{"tool":"write_file","args":{"path":"a.py","content":"x","summary":"s"}}</agentic_call>\n'
        '<agentic_call>{"tool":"run","args":{"cmd":"python -m pytest"}}</agentic_call>'
    )
    assert [c.name for c in extract_all_tool_calls(txt)] == ["write_file", "run"]


def test_legacy_tool_call_tag_still_parsed():
    # <tool_call> is the reserved native delimiter Ollama's own qwen renderer/
    # parser intercepts server-side (see llm/tool_parser.py module docstring) —
    # the Worker is never prompted to use it, but the parser stays tolerant of
    # it turning up (e.g. a model reverting to its native training, old dumps).
    txt = '<tool_call>{"tool":"read_file","args":{"path":"a.py"}}</tool_call>'
    c = extract_tool_call(txt)
    assert c is not None and c.name == "read_file" and c.args["path"] == "a.py"


def test_mismatched_tag_pair_falls_back_to_bare_json():
    # An <agentic_call> opener must close with </agentic_call>, not </tool_call>
    # (and vice versa) — no cross-tag matching in the tag-block strategy. The
    # bare JSON object inside is still recovered by the bare-JSON-in-prose
    # fallback strategy, so the call isn't lost.
    txt = '<agentic_call>{"tool":"read_file","args":{"path":"a.py"}}</tool_call>'
    c = extract_tool_call(txt)
    assert c is not None and c.name == "read_file" and c.args["path"] == "a.py"


def test_salvage_unescaped_quotes_and_newlines():
    # content holds REAL newlines and UNESCAPED quotes -> not valid JSON; must be salvaged.
    txt = '{"tool": "write_file", "args": {"path": "m.py", "content": "def f():\n    print("hi")\n", "summary": "m"}}'
    calls = extract_all_tool_calls(txt)
    assert len(calls) == 1
    c = calls[0]
    assert c.name == "write_file"
    assert c.salvaged is True
    assert c.args["path"] == "m.py"
    assert 'print("hi")' in c.args["content"]


def test_trailing_comma_and_fence():
    c = extract_tool_call('```json\n{"tool":"read_file","args":{"path":"x.py",}}\n```')
    assert c is not None and c.name == "read_file" and c.args["path"] == "x.py"


def test_extract_json_picks_richest_object():
    txt = 'noise {"a":1} then {"project":"p","tasks":[{"id":"T1"}]} tail'
    data = extract_json(txt)
    assert isinstance(data, dict) and data.get("project") == "p"


def test_unknown_tool_is_flagged_not_dropped():
    c = extract_tool_call('<agentic_call>{"tool":"frobnicate","args":{}}</agentic_call>')
    assert c is not None and not c.is_known
