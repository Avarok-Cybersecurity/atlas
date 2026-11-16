#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Dev-only Jinja2 oracle for the embedded Sharp chat template (lane L39).

Writes ``cases.json`` (the conversations and render options) and ``goldens.json``
(what Python Jinja2 renders for each) next to the Rust test that checks Atlas's
minijinja rendering against them byte for byte
(``crates/spark-server/src/tokenizer/tests/sharp_template.rs``).

Why a Python oracle: the point of the golden is to catch a place where Atlas's
renderer (minijinja plus its compatibility layer) differs from the Jinja2 the
template was written for. Rendering the expectation with Atlas itself would hide
exactly that. This script is NOT part of the shipped binary and is never run by
the build; run it only when the embedded template or the case list changes:

    python3 -I tools/chat_template_goldens/gen_sharp_goldens.py          # rewrite both files
    python3 -I tools/chat_template_goldens/gen_sharp_goldens.py --check  # exit 1 if they are stale

Requires ``jinja2`` (3.1.x). It does not import or run anything from the
upstream template repository: the template text comes from the embedded asset,
and the cases are written here.

The Jinja2 environment mirrors Atlas's (``tokenizer/jinja_helpers.rs``):
``trim_blocks`` and ``lstrip_blocks`` on, lenient undefined variables,
``raise_exception``, and a ``tojson`` that is serde_json compact output with
minijinja's HTML-safe escaping of ``< > & '`` (``\\u003c`` and so on), because
Atlas overrides Python's key-sorting ``tojson``. The render context mirrors
``tokenizer/chat_render.rs``: ``add_generation_prompt`` true, ``add_vision_id``
false, ``reasoning_effort`` defaulting to ``medium`` when thinking is on and
``none`` when it is off, and ``tools``, ``preserve_thinking`` and ``terse``
present only when the case gives them.

Cases avoid what Atlas rewrites before rendering (a ``developer`` role, inline
``<|think_on|>``/``<|think_off|>`` tags, an unclosed ``<think>`` ahead of a
``<tool_call>``, tool arguments that are JSON text), so the Python render sees
the same messages the Rust one does.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import jinja2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ASSET = REPO / "crates/spark-server/assets/chat_templates/qwen-sharp-v22.5.0.jinja"
DATA = REPO / "crates/spark-server/test_data/chat_templates/sharp"
NO_TERSE_PREFIX = "{%- set terse = false %}\n"
FUZZ_SEED = 20261112
FUZZ_CASES = 36


# --------------------------------------------------------------------------------------------------------
# The oracle environment
# --------------------------------------------------------------------------------------------------------
class RaisedException(jinja2.TemplateError):
    """What the template's ``raise_exception`` throws."""


def raise_exception(message):
    raise RaisedException(message)


def atlas_tojson(value, indent=None):
    """serde_json compact output plus minijinja's escaping of < > & ' (see the module doc)."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    for char, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("'", "\\u0027")):
        text = text.replace(char, escaped)
    return text


def make_env():
    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.Undefined,
        keep_trailing_newline=False,
        autoescape=False,
    )
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = atlas_tojson
    return env


def template_source(variant):
    text = ASSET.read_text(encoding="utf-8")
    if variant == "sharp":
        return text
    if variant == "sharp-no-terse":
        return NO_TERSE_PREFIX + text
    raise ValueError("unknown variant %r" % variant)


def build_context(case):
    thinking = case["enable_thinking"]
    effort = case.get("reasoning_effort") or ("medium" if thinking else "none")
    ctx = {
        "messages": case["messages"],
        "add_generation_prompt": True,
        "enable_thinking": thinking,
        "reasoning_effort": effort,
        "disable_tool_steering": bool(case.get("disable_tool_steering", False)),
        "add_vision_id": False,
    }
    if "tools" in case:
        ctx["tools"] = case["tools"]
    if "preserve_thinking" in case:
        ctx["preserve_thinking"] = case["preserve_thinking"]
    if "terse" in case:
        ctx["terse"] = case["terse"]
    return ctx


def render_case(env, case, cache={}):
    variant = case.get("variant", "sharp")
    if variant not in cache:
        cache[variant] = env.from_string(template_source(variant))
    try:
        return cache[variant].render(**build_context(case))
    except RaisedException as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------------------------------------
def u(text):
    return {"role": "user", "content": text}


def s(text):
    return {"role": "system", "content": text}


def a(text, reasoning=None, tool_calls=None, key="reasoning_content"):
    msg = {"role": "assistant", "content": text}
    if reasoning is not None:
        msg[key] = reasoning
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def t(text):
    return {"role": "tool", "content": text}


def call(name, arguments, call_id=None):
    return {"id": call_id or "call_" + name, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_def(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


WEATHER = tool_def(
    "get_weather",
    "Look up the current weather for a city.",
    {"city": {"type": "string", "description": "City name."}, "days": {"type": "integer", "description": "Horizon in days."}},
    ["city", "days"],
)
SHELL = tool_def(
    "bash",
    "Run a shell command; use 'a && b' chains and <file> redirection with care.",
    {"command": {"type": "string", "description": "The command, e.g. echo 'hi' > <out>."}},
    ["command"],
)
TOOLS = [WEATHER]
TOOLS2 = [WEATHER, SHELL]


def case(name, messages, *, variant="sharp", tools=None, thinking=True, effort=None, preserve=None, terse=None,
         steering=None):
    out = {"name": name, "variant": variant, "messages": messages, "enable_thinking": thinking}
    if tools is not None:
        out["tools"] = tools
    if effort is not None:
        out["reasoning_effort"] = effort
    if preserve is not None:
        out["preserve_thinking"] = preserve
    if terse is not None:
        out["terse"] = terse
    if steering is not None:
        out["disable_tool_steering"] = steering
    return out


def hand_cases():
    cs = []
    add = cs.append
    # Single turn, with and without a system prompt, thinking on and off.
    add(case("single_user_on", [u("What is 17 times 23?")]))
    add(case("single_user_off", [u("What is 17 times 23?")], thinking=False))
    add(case("system_user_on", [s("You are a careful assistant."), u("Name the capital of Peru.")]))
    add(case("system_user_off", [s("You are a careful assistant."), u("Name the capital of Peru.")], thinking=False))
    add(case("two_system_messages_merge", [s("Rule one."), s("Rule two."), u("Go.")]))
    add(case("system_after_user_stays_in_order", [u("Hi"), s("Late system note."), u("Again")]))
    # The terse knob and the no-terse variant.
    add(case("terse_false_with_system", [s("Be brief."), u("Why is the sky blue?")], terse=False))
    add(case("terse_false_no_system", [u("Why is the sky blue?")], terse=False))
    add(case("terse_true_no_system", [u("Why is the sky blue?")], terse=True))
    add(case("no_terse_variant_with_system", [s("Be brief."), u("Why is the sky blue?")], variant="sharp-no-terse"))
    add(case("no_terse_variant_ignores_kwarg", [u("Why is the sky blue?")], variant="sharp-no-terse", terse=True))
    # Effort handling.
    for effort in ("low", "minimal", "high", "xhigh", "max", "medium", "none", "bogus"):
        add(case("effort_" + effort, [u("Plan a three day trip to Lisbon.")], effort=effort))
    add(case("effort_low_thinking_off", [u("Plan a trip.")], thinking=False, effort="low"))
    add(case("effort_high_with_tools", [u("Weather in Oslo for 2 days?")], tools=TOOLS, effort="high"))
    # Multi-turn with reasoning kept in history.
    two_turns = [
        u("Add 2 and 3."),
        a("The sum is 5.", reasoning="2 plus 3 is 5."),
        u("Now multiply that by 4."),
    ]
    add(case("multi_turn_reasoning_content_on", two_turns))
    add(case("multi_turn_reasoning_content_off", two_turns, thinking=False))
    add(case("multi_turn_reasoning_preserve_false", two_turns, preserve=False))
    add(case("multi_turn_reasoning_preserve_true", two_turns, preserve=True))
    add(case("multi_turn_reasoning_key_thinking", [u("Hi"), a("Hello.", reasoning="greet back", key="thinking"), u("Bye")]))
    add(case("multi_turn_reasoning_key_reasoning", [u("Hi"), a("Hello.", reasoning="greet back", key="reasoning"), u("Bye")]))
    add(case("multi_turn_no_reasoning", [u("Hi"), a("Hello."), u("Bye")]))
    add(case("multi_turn_empty_reasoning", [u("Hi"), a("Hello.", reasoning=""), u("Bye")]))
    add(case("multi_turn_three_turns_default_retention", [
        u("One"), a("A1", reasoning="r1"), u("Two"), a("A2", reasoning="r2"), u("Three"), a("A3", reasoning="r3"), u("Four"),
    ]))
    add(case("multi_turn_three_turns_preserve_false", [
        u("One"), a("A1", reasoning="r1"), u("Two"), a("A2", reasoning="r2"), u("Three"),
    ], preserve=False))
    # Reasoning that arrives inline in the assistant text (startswith/endswith paths).
    add(case("inline_think_block", [u("Q1"), a("<think>\nplan the answer\n</think>\n\nAnswer one."), u("Q2")]))
    add(case("inline_thinking_block", [u("Q1"), a("<thinking>\nplan the answer\n</thinking>\n\nAnswer one."), u("Q2")]))
    add(case("inline_leading_close_think", [u("Q1"), a("</think>\n\nAnswer after a prefilled think."), u("Q2")]))
    add(case("inline_close_mid_content", [u("Q1"), a("reasoning text\n</think>\n\nAnswer."), u("Q2")]))
    add(case("inline_think_plus_explicit_reasoning", [
        u("Q1"), a("<think>\nold\n</think>\n\nAnswer.", reasoning="explicit wins"), u("Q2"),
    ]))
    add(case("inline_think_only_open", [u("Q1"), a("<think>\nnever closed, no tool call"), u("Q2")]))
    # Tools.
    add(case("tools_on", [u("Weather in Oslo for 3 days?")], tools=TOOLS))
    add(case("tools_off", [u("Weather in Oslo for 3 days?")], tools=TOOLS, thinking=False))
    add(case("tools_with_system", [s("Use tools when useful."), u("Weather in Oslo for 3 days?")], tools=TOOLS2))
    add(case("tools_steering_disabled", [u("Weather in Oslo?")], tools=TOOLS, steering=True))
    add(case("tools_no_terse_variant", [u("Weather in Oslo?")], tools=TOOLS, variant="sharp-no-terse"))
    add(case("tools_terse_false", [s("Be exact."), u("Weather in Oslo?")], tools=TOOLS, terse=False))
    add(case("tools_stood_down_by_runtime_marker", [s("[TOOL_REQUEST] runtime protocol"), u("Weather?")], tools=TOOLS))
    add(case("tools_empty_list", [u("Hello")], tools=[]))
    roundtrip = [
        u("Weather in Oslo for 3 days?"),
        a("", reasoning="I should call the tool.", tool_calls=[call("get_weather", {"city": "Oslo", "days": 3})]),
        t("{\"temp_c\": 4, \"sky\": \"overcast\"}"),
    ]
    add(case("tool_roundtrip_on", roundtrip, tools=TOOLS))
    add(case("tool_roundtrip_off", roundtrip, tools=TOOLS, thinking=False))
    add(case("tool_roundtrip_then_user", roundtrip + [a("It is 4 C and overcast.", reasoning="report"), u("And Bergen?")], tools=TOOLS))
    add(case("tool_call_with_content", [
        u("Weather in Oslo?"),
        a("Let me check.", tool_calls=[call("get_weather", {"city": "Oslo", "days": 1})]),
        t("sunny"),
    ], tools=TOOLS))
    add(case("tool_call_empty_content_no_reasoning", [
        u("Run it"), a("", tool_calls=[call("bash", {"command": "ls"})]), t("a.txt"),
    ], tools=TOOLS2))
    add(case("tool_parallel_calls", [
        u("Weather in Oslo and Rome?"),
        a("", reasoning="two calls", tool_calls=[
            call("get_weather", {"city": "Oslo", "days": 1}, "c1"),
            call("get_weather", {"city": "Rome", "days": 1}, "c2"),
        ]),
        t("Oslo: 4 C"),
        t("Rome: 17 C"),
    ], tools=TOOLS))
    add(case("tool_arguments_special_values", [
        u("Run it"),
        a("", tool_calls=[call("bash", {"command": "echo 'a & b' > <out>", "flags": [1, 2, 3], "verbose": True,
                                          "limit": None, "opts": {"x": "y"}})]),
        t("done"),
    ], tools=TOOLS2))
    add(case("tool_arguments_raw_text", [
        u("Run it"), a("", tool_calls=[call("bash", "raw argument text, not JSON")]), t("ok"),
    ], tools=TOOLS2))
    add(case("tool_arguments_missing", [
        u("Run it"), a("", tool_calls=[{"id": "c", "type": "function", "function": {"name": "bash"}}]), t("ok"),
    ], tools=TOOLS2))
    add(case("tool_error_exception_prefix", [
        u("Run it"), a("", tool_calls=[call("bash", {"command": "x"})]), t("Exception: boom"),
    ], tools=TOOLS2))
    add(case("tool_error_failed_to_prefix", [
        u("Run it"), a("", tool_calls=[call("bash", {"command": "x"})]), t("Failed to open the file"),
    ], tools=TOOLS2))
    add(case("tool_error_twice_escalates", [
        u("Run it"),
        a("", tool_calls=[call("bash", {"command": "x"})]), t("Exception: first"),
        a("", tool_calls=[call("bash", {"command": "y"})]), t("Traceback (most recent call last):\n  boom"),
    ], tools=TOOLS2))
    add(case("tool_error_then_success_resets", [
        u("Run it"),
        a("", tool_calls=[call("bash", {"command": "x"})]), t("Exception: first"),
        a("", tool_calls=[call("bash", {"command": "y"})]), t("all good"),
    ], tools=TOOLS2))
    add(case("tool_weak_error_short", [
        u("Run it"), a("", tool_calls=[call("bash", {"command": "x"})]), t("error: file missing"),
    ], tools=TOOLS2))
    add(case("tool_weak_error_suppressed_long", [
        u("Run it"), a("", tool_calls=[call("bash", {"command": "x"})]),
        t("error: file missing " + "padding line of output. " * 30),
    ], tools=TOOLS2))
    add(case("tool_result_looks_like_code", [
        u("Show it"), a("", tool_calls=[call("bash", {"command": "cat a.py"})]), t("def main():\n    raise error('x')"),
    ], tools=TOOLS2))
    add(case("tool_exit_code_nonzero", [
        u("Run it"), a("", tool_calls=[call("bash", {"command": "false"})]), t("exit code: 1\nnothing"),
    ], tools=TOOLS2))
    add(case("tool_error_field_null_is_ok", [
        u("Run it"), a("", tool_calls=[call("bash", {"command": "x"})]), t("{\"error\": null, \"value\": 3}"),
    ], tools=TOOLS2))
    add(case("user_message_is_tool_response", [
        u("Weather?"),
        a("", tool_calls=[call("get_weather", {"city": "Oslo", "days": 1})]),
        u("<tool_response>\n4 C\n</tool_response>"),
    ], tools=TOOLS))
    # Content forms.
    add(case("content_parts_text", [{"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": " second"}]}]))
    add(case("content_part_image", [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]))
    add(case("content_part_two_images_then_video", [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"}, {"type": "video"}, {"type": "text", "text": "Compare."},
    ]}]))
    add(case("content_empty_string", [u(""), a("ok"), u("again")]))
    add(case("content_whitespace_trimmed", [u("   padded   "), a("  reply  ", reasoning="  why  "), u("\nnext\n")]))
    add(case("unicode_content", [u("Vad är huvudstaden i Sverige? 你好 🙂 café"), a("Stockholm.", reasoning="svar på svenska"), u("Tack ✓")]))
    add(case("ends_with_assistant_prefill", [u("Continue this"), a("Once upon a")]))
    add(case("assistant_first", [a("Hello, how can I help?"), u("Weather?")]))
    add(case("unknown_role_is_wrapped", [{"role": "narrator", "content": "It was dark."}, u("Go on")]))
    # Errors the template raises on purpose.
    add(case("error_image_in_system", [{"role": "system", "content": [{"type": "image"}]}, u("hi")]))
    add(case("error_video_in_system", [{"role": "system", "content": [{"type": "video"}]}, u("hi")]))
    add(case("error_unexpected_content_type", [{"role": "user", "content": 5}]))
    add(case("error_unexpected_item", [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}]))
    add(case("error_no_messages", []))
    return cs


# --------------------------------------------------------------------------------------------------------
# Deterministic generated conversations
# --------------------------------------------------------------------------------------------------------
USER_BANK = [
    "Summarize the plot of Hamlet in two sentences.",
    "How many ounces are in 3.5 pounds?",
    "Write a haiku about autumn rain.",
    "Explain the difference between TCP and UDP.",
    "Vad är skillnaden mellan en tråd och en process?",
    "Translate 'good morning' into Japanese: おはよう is right, correct?",
    "List three uses of a hash map.",
    "Is 91 prime? Show the check.",
]
ASSIST_BANK = [
    "Here is the short version.",
    "It is 56 ounces.",
    "Autumn rain taps the roof,\nthe lamp hums back softly.",
    "TCP is reliable and ordered; UDP is not.",
    "A process owns memory; a thread shares it.",
    "No, 91 = 7 x 13.",
]
REASON_BANK = [
    "The user wants a brief answer; keep it lean.",
    "Check the arithmetic: 3.5 times 16 is 56.",
    "Consider both protocols' guarantees first.",
    "Seven times thirteen is ninety-one.",
]
TOOL_OUT_BANK = [
    "{\"temp_c\": 12, \"sky\": \"clear\"}",
    "file1.txt\nfile2.txt",
    "Exception: connection refused",
    "Failed to resolve host",
    "ok",
    "Traceback (most recent call last):\n  File x, line 1",
    "42",
    "error: not found",
]
ARG_BANK = [
    {"city": "Oslo", "days": 2},
    {"command": "ls -la"},
    {"city": "Zürich", "days": 5},
    {"command": "grep -n 'a|b' file.txt"},
]
EFFORTS = [None, None, "low", "medium", "high", "none", "max"]


def fuzz_cases():
    rng = random.Random(FUZZ_SEED)
    cs = []
    for i in range(FUZZ_CASES):
        use_tools = rng.random() < 0.5
        msgs = []
        if rng.random() < 0.35:
            msgs.append(s(rng.choice(["You are a helpful assistant.", "Answer in plain prose.", "Be precise."])))
        for turn in range(rng.randint(1, 4)):
            msgs.append(u(rng.choice(USER_BANK)))
            last_turn = turn == 3 or rng.random() < 0.4
            if last_turn and rng.random() < 0.5:
                break
            roll = rng.random()
            if use_tools and roll < 0.45:
                n_calls = rng.choice([1, 1, 2])
                calls = [call(rng.choice(["get_weather", "bash"]), rng.choice(ARG_BANK), "c%d_%d" % (i, k)) for k in range(n_calls)]
                msgs.append(a(rng.choice(["", "", rng.choice(ASSIST_BANK)]),
                              reasoning=rng.choice([None, rng.choice(REASON_BANK)]), tool_calls=calls))
                for _ in calls:
                    msgs.append(t(rng.choice(TOOL_OUT_BANK)))
                if rng.random() < 0.5:
                    msgs.append(a(rng.choice(ASSIST_BANK), reasoning=rng.choice([None, rng.choice(REASON_BANK)])))
            elif roll < 0.75:
                msgs.append(a(rng.choice(ASSIST_BANK), reasoning=rng.choice([None, rng.choice(REASON_BANK)])))
            elif roll < 0.9:
                msgs.append(a("<think>\n%s\n</think>\n\n%s" % (rng.choice(REASON_BANK), rng.choice(ASSIST_BANK))))
            else:
                msgs.append(a(rng.choice(ASSIST_BANK), reasoning=rng.choice(REASON_BANK), key=rng.choice(["thinking", "reasoning"])))
        while msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls") and rng.random() < 0.9:
            msgs.append(u(rng.choice(USER_BANK)))
        out = case(
            "fuzz_%02d" % i,
            msgs,
            variant=rng.choice(["sharp", "sharp", "sharp-no-terse"]),
            tools=(TOOLS2 if use_tools else None),
            thinking=rng.random() < 0.6,
            effort=rng.choice(EFFORTS),
            preserve=rng.choice([None, None, True, False]),
            terse=rng.choice([None, None, True, False]),
        )
        cs.append(out)
    return cs


# --------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------
def dump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


def build():
    template_bytes = ASSET.read_bytes()
    cases = hand_cases() + fuzz_cases()
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate case names")
    env = make_env()
    renders = {c["name"]: render_case(env, c) for c in cases}
    cases_doc = {
        "format": 1,
        "note": "Generated by tools/chat_template_goldens/gen_sharp_goldens.py. Do not edit by hand.",
        "cases": cases,
    }
    goldens_doc = {
        "format": 1,
        "note": "Python Jinja2 renders of cases.json. Regenerate only when the embedded template or the cases change.",
        "jinja2": jinja2.__version__,
        "template_bytes": len(template_bytes),
        "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "renders": renders,
    }
    return dump(cases_doc), dump(goldens_doc)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if the committed files are stale")
    args = ap.parse_args(argv)
    cases_text, goldens_text = build()
    targets = {DATA / "cases.json": cases_text, DATA / "goldens.json": goldens_text}
    if args.check:
        stale = [p.name for p, text in targets.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            print("stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("up to date")
        return 0
    DATA.mkdir(parents=True, exist_ok=True)
    for path, text in targets.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print("wrote %s (%d bytes)" % (path.relative_to(REPO), len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
