#!/usr/bin/env python3
"""
test_long_context.py — Test 35B model output quality at 12k+ token context.

Downloads a public-domain text (Alice in Wonderland) from Project Gutenberg,
trims to target ISL, and sends it as a chat completion with a summarization
instruction. Checks that the response is coherent (not garbage).

Usage:
  python test_long_context.py [--url URL] [--model MODEL] [--isl N]
  python test_long_context.py --isl 12000   # ~12k token input
  python test_long_context.py --isl 16000   # ~16k token input
  python test_long_context.py --sweep        # test 512,1024,2048,4096,8192,12000,16000
"""

import argparse, json, sys, time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DEFAULT_URL   = "http://localhost:8888"
DEFAULT_MODEL = "m"
GUTENBERG_URL = "https://www.gutenberg.org/files/11/11-0.txt"

# Fallback: Paul Graham's "What I Worked On" essay
FALLBACK_URL  = "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/needlehaystack/PaulGrahamEssays/superlinear.txt"


def fetch_long_text() -> str:
    """Fetch a long public-domain text for context window testing."""
    for url in [GUTENBERG_URL, FALLBACK_URL]:
        try:
            resp = urlopen(url, timeout=30)
            text = resp.read().decode("utf-8", errors="replace")
            # Strip Project Gutenberg header/footer
            if "*** START" in text:
                text = text.split("*** START")[1]
                if "***" in text:
                    text = text.split("***", 1)[1]
            if "*** END" in text:
                text = text.split("*** END")[0]
            return text.strip()
        except Exception as e:
            print(f"  Warning: {url} failed: {e}", file=sys.stderr)
    # Final fallback: generate synthetic text
    print("  Using synthetic text (all URLs failed)", file=sys.stderr)
    words = (
        "The quick brown fox jumped over the lazy dog near a river bank. "
        "Mountains rise above the clouds while birds sing their morning songs. "
        "Science explores the universe through careful observation and experiment. "
        "Ancient civilizations built remarkable structures that still stand today. "
        "Music fills the air with rhythm and harmony across every culture. "
        "Technology advances rapidly changing how people communicate and work. "
        "Forests provide shelter for countless species of plants and animals. "
        "Ocean waves crash upon the shore under the light of the moon. "
    )
    return (words * 5000)[:200000]


def trim_to_tokens(text: str, target_tokens: int) -> str:
    """Approximate trim: ~1.3 words per token for English text."""
    words = text.split()
    # Rough heuristic: 1 token ≈ 0.75 words for BPE
    target_words = int(target_tokens * 0.75)
    if len(words) > target_words:
        words = words[:target_words]
    return " ".join(words)


def send_request(url: str, model: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send a non-streaming chat completion and return the response."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()

    req = Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    resp = urlopen(req, timeout=600)
    data = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - t0
    return data, elapsed


def check_quality(text: str) -> tuple[bool, str]:
    """
    Heuristic quality check: detect garbage output.

    Garbage indicators:
    - High ratio of non-ASCII characters
    - Excessive repetition (same 3-gram repeated 5+ times)
    - Very short response (< 20 chars) for a summarization task
    - All-caps or all-punctuation
    """
    if len(text) < 20:
        return False, "Response too short"

    # Non-ASCII ratio
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) > 0.3:
        return False, f"High non-ASCII ratio: {non_ascii}/{len(text)} ({non_ascii/len(text):.0%})"

    # Repetition check: find most common 3-gram
    words = text.split()
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        from collections import Counter
        common = Counter(trigrams).most_common(1)
        if common and common[0][1] >= max(5, len(trigrams) * 0.15):
            return False, f"Excessive repetition: '{common[0][0]}' repeated {common[0][1]}x"

    # Check for coherent English words
    english_words = {"the", "a", "is", "are", "was", "in", "of", "to", "and", "that", "it", "for", "on", "with"}
    lower_words = set(w.lower().strip(".,!?;:\"'()") for w in words)
    overlap = lower_words & english_words
    if len(words) > 10 and len(overlap) < 2:
        return False, "No common English words found — likely garbage"

    return True, "OK"


def test_isl(url: str, model: str, text: str, isl: int) -> tuple[bool, str]:
    """Test a single ISL value."""
    trimmed = trim_to_tokens(text, isl)
    word_count = len(trimmed.split())

    prompt = (
        f"Below is a passage of text. Please provide a brief 3-sentence summary "
        f"of the main topics covered.\n\n{trimmed}"
    )

    print(f"  ISL ~{isl:>6} ({word_count:>6} words) ... ", end="", flush=True)

    try:
        data, elapsed = send_request(url, model, prompt, max_tokens=256)
    except HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else str(e)
        print(f"HTTP {e.code}: {body[:100]}")
        return False, f"HTTP error {e.code}"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, str(e)

    choice = data.get("choices", [{}])[0]
    response_text = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")
    tps = usage.get("response_token/s", "?")

    ok, reason = check_quality(response_text)
    status = "PASS" if ok else "FAIL"

    print(f"{status}  ptok={prompt_tokens}  ctok={completion_tokens}  "
          f"tps={tps}  e2e={elapsed:.1f}s")
    if not ok:
        print(f"    Reason: {reason}")
        print(f"    Response: {response_text[:200]}...")
    else:
        print(f"    Response: {response_text[:150]}...")

    return ok, reason


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url",   default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--isl",   type=int, default=12000,
                    help="Target input sequence length in tokens")
    ap.add_argument("--sweep", action="store_true",
                    help="Test multiple ISL values: 512,1024,2048,4096,8192,12000,16000")
    args = ap.parse_args()

    # Health check
    try:
        urlopen(f"{args.url}/health", timeout=5).read()
    except Exception as e:
        print(f"ERROR: server not reachable at {args.url}: {e}", file=sys.stderr)
        sys.exit(1)

    print("Fetching long text from Project Gutenberg...")
    text = fetch_long_text()
    total_words = len(text.split())
    print(f"  Got {total_words} words (~{int(total_words/0.75)} estimated tokens)")
    print()

    isls = [512, 1024, 2048, 4096, 8192, 12000, 16000] if args.sweep else [args.isl]

    print(f"Long Context Quality Test")
    print(f"  URL:   {args.url}")
    print(f"  Model: {args.model}")
    print(f"  ISLs:  {isls}")
    print()

    results = {}
    for isl in isls:
        ok, reason = test_isl(args.url, args.model, text, isl)
        results[isl] = (ok, reason)
        time.sleep(1)

    print()
    print("=" * 60)
    print("Summary:")
    all_pass = True
    for isl, (ok, reason) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  ISL {isl:>6}: {status}  {reason}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll tests passed — no garbage output detected.")
    else:
        print("\nSome tests FAILED — garbage output detected at long context!")
        # Identify the threshold
        passing = [isl for isl, (ok, _) in results.items() if ok]
        failing = [isl for isl, (ok, _) in results.items() if not ok]
        if passing and failing:
            print(f"  Last passing ISL: {max(passing)}")
            print(f"  First failing ISL: {min(failing)}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
