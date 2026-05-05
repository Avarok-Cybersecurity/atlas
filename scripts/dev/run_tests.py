#!/usr/bin/env python3
"""Atlas model test suite.

Usage:
  python3 run_tests.py <model_id> <port> [--long-context] [--speculative] [extra_args...]

Flags:
  --long-context   Run NIAH (needle-in-a-haystack) tests at 4K-64K context
  --speculative    Enable MTP speculative decoding
  Other args are passed directly to the Atlas server CLI.
"""
import sys, json, time, subprocess, re, signal, urllib.request

# ── Parse args ──────────────────────────────────────────────────
MODEL = sys.argv[1]
PORT = int(sys.argv[2])
raw_extra = sys.argv[3:] if len(sys.argv) > 3 else []
LONG_CONTEXT = "--long-context" in raw_extra
raw_extra = [a for a in raw_extra if a != "--long-context"]
EXTRA = " ".join(raw_extra)

IMAGE = "atlas-gb10:latest"
CONTAINER = f"atlas-test-{PORT}"
HF_CACHE = "/workspace/.cache/huggingface/hub"

is_mistral = "mistral" in MODEL.lower()
is_nano = "nano" in MODEL.lower()

MAX_SEQ_LEN = 65536 if LONG_CONTEXT else 16384

# ── Helpers ─────────────────────────────────────────────────────
def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

def cleanup():
    run(f"sudo docker stop {CONTAINER} 2>/dev/null")
    run(f"sudo docker rm {CONTAINER} 2>/dev/null")

signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(1)))

# ── Start container ─────────────────────────────────────────────
cleanup()
extra_server = EXTRA
if LONG_CONTEXT and "--kv-high-precision-layers" not in extra_server:
    extra_server += " --kv-high-precision-layers auto"

cmd = (f"sudo docker run -d --name {CONTAINER} "
       f"--gpus all --ipc=host --network host "
       f"-v {HF_CACHE}:/root/.cache/huggingface/hub "
       f"{IMAGE} serve {MODEL} --port {PORT} --max-seq-len {MAX_SEQ_LEN} {extra_server}")
r = run(cmd)
if r.returncode != 0:
    print(json.dumps({"model": MODEL, "status": "LAUNCH_FAIL", "error": r.stderr[:200]}))
    sys.exit(1)

# ── Wait for ready ──────────────────────────────────────────────
print(f"Loading {MODEL} (max_seq_len={MAX_SEQ_LEN})...", file=sys.stderr)
for i in range(90):  # 7.5 min (64K needs more KV cache alloc time)
    time.sleep(5)
    logs = run(f"sudo docker logs {CONTAINER} 2>&1").stdout
    if "Listening on" in logs:
        break
    if "Error:" in logs or "panic" in logs:
        err = [l for l in logs.split("\n") if "Error:" in l or "panic" in l][-1:]
        print(json.dumps({"model": MODEL, "status": "CRASH", "error": str(err[:200])}))
        cleanup(); sys.exit(1)
else:
    print(json.dumps({"model": MODEL, "status": "TIMEOUT"}))
    cleanup(); sys.exit(1)

# ── Test runner ─────────────────────────────────────────────────
def test(name, prompt, max_tokens=150, pattern=None, timeout_s=120):
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            r = json.loads(resp.read())
        c = r["choices"][0]
        u = r["usage"]
        content = c["message"]["content"]
        passed = bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL)) if pattern else True
        return {
            "test": name,
            "status": "PASS" if passed else "FAIL",
            "content": content[:200],
            "tokens": u.get("completion_tokens", 0),
            "prompt_tokens": u.get("prompt_tokens", 0),
            "tok_s": round(u.get("response_token/s", 0), 1),
            "ttft_ms": round(u.get("time_to_first_token_ms", 0), 1),
            "finish": c.get("finish_reason", ""),
        }
    except Exception as e:
        return {"test": name, "status": "ERROR", "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════
# Standard 6-test quality suite
# ═══════════════════════════════════════════════════════════════
results = []
print(f"Testing {MODEL}...", file=sys.stderr)

results.append(test("capital", "What is the capital of France?", 50, r"Paris"))

results.append(test("fibonacci", "Write the first 10 Fibonacci numbers separated by commas.", 80,
                     r"0.*1.*1.*2.*3.*5.*8.*13.*21.*34"))

if is_mistral:
    results.append(test("count",
        "Count from one to ten using words, separated by commas.",
        60, r"one.*two.*three.*four.*five.*six.*seven.*eight.*nine.*ten"))
else:
    results.append(test("count", "Count from 1 to 20, separated by commas.", 100,
                         r"1.*2.*3.*4.*5.*6.*7.*8.*9.*10.*11.*12.*13.*14.*15.*16.*17.*18.*19.*20"))

results.append(test("coherence", "Explain what photosynthesis is in 2-3 sentences.", 150,
                     r"sunlight|chloro|oxygen|carbon|glucose|plant"))

if is_mistral:
    results.append(test("code",
        "What is a prime number? Name two examples.",
        60, r"(prime|divis).*(2|3|5|7|11|13)"))
else:
    results.append(test("code", "Write a Python function called is_prime that returns True if a number is prime.", 200,
                         r"def is_prime"))

results.append(test("planets", "List all 8 planets in our solar system in order from the sun.", 200,
                     r"Mercury.*Venus.*Earth.*Mars.*Jupiter.*Saturn.*Uranus.*Neptune"))


# ═══════════════════════════════════════════════════════════════
# Long-context NIAH (needle-in-a-haystack) tests
# ═══════════════════════════════════════════════════════════════
if LONG_CONTEXT:
    print("Running NIAH long-context tests...", file=sys.stderr)

    NEEDLE = "IMPORTANT FACT: The secret project codename is AURORA-VELVET-7742."
    NEEDLE_QUESTION = (
        "\n\nBased on the text above, what is the secret project codename? "
        "Reply with ONLY the codename, nothing else."
    )
    NEEDLE_PATTERN = r"AURORA.VELVET.7742"

    # 10 diverse filler paragraphs (~150-200 words each, ~200-250 tokens)
    FILLER = [
        "The history of cartography stretches back thousands of years. Ancient civilizations in Mesopotamia created clay tablet maps as early as 2300 BCE. The Greeks made significant advances, with Eratosthenes calculating the Earth's circumference with remarkable accuracy around 240 BCE. During the Age of Exploration, European cartographers produced increasingly detailed world maps, though many contained speculative elements for uncharted territories. The Mercator projection, introduced in 1569, became the standard for nautical navigation despite its well-known distortion of area near the poles. Modern cartography has been revolutionized by satellite imagery, GPS technology, and geographic information systems that can layer multiple data types onto a single interactive map. Digital mapping services now process billions of queries daily.",

        "The process of fermentation has been utilized by humans for millennia. Ancient Egyptians discovered that allowing grain mixtures to sit produced an alcoholic beverage, while simultaneously developing leavened bread through similar microbial processes. Louis Pasteur's groundbreaking work in the 1850s revealed that fermentation was caused by living microorganisms rather than spontaneous chemical reactions. Today, industrial fermentation produces a vast range of products beyond food and drink, including antibiotics, biofuels, organic acids, and enzymes. The global fermentation technology market continues to grow as researchers discover new applications in pharmaceuticals, agriculture, and sustainable materials production.",

        "Coral reefs are among the most biodiverse ecosystems on Earth, often called the rainforests of the sea. Despite covering less than one percent of the ocean floor, they support approximately twenty-five percent of all marine species. The reef structure itself is built by tiny coral polyps that secrete calcium carbonate over thousands of years. These ecosystems face unprecedented threats from ocean acidification, rising sea temperatures, and pollution, with scientists estimating that up to ninety percent of reefs could disappear by 2050 under current climate projections. Conservation efforts include coral gardening, marine protected areas, and research into heat-resistant coral strains that might survive warming oceans.",

        "The development of semiconductor technology transformed the twentieth century. The invention of the transistor at Bell Labs in 1947 replaced bulky vacuum tubes and enabled the miniaturization of electronic circuits. Gordon Moore's famous observation in 1965 that transistor density doubles approximately every two years guided industry investment for decades. Modern processors contain billions of transistors etched at scales below five nanometers. The semiconductor supply chain spans multiple continents, with specialized fabrication facilities costing upwards of twenty billion dollars to construct. Emerging technologies like quantum computing and neuromorphic chips may eventually supplement or surpass traditional silicon-based architectures.",

        "The Amazon rainforest spans approximately five and a half million square kilometers across nine South American countries. It produces roughly twenty percent of the world's oxygen and contains approximately ten percent of all known species. Indigenous communities have inhabited the region for at least eleven thousand years, developing sophisticated knowledge of medicinal plants and sustainable land management practices. Deforestation rates have fluctuated significantly over recent decades, driven by agricultural expansion, logging, and mining activities. International conservation programs work alongside local communities to preserve this critical biome through protected areas, sustainable forestry certifications, and payment for ecosystem services.",

        "The invention of the printing press by Johannes Gutenberg around 1440 fundamentally altered the course of human civilization. Before movable type, books were painstakingly copied by hand, making them extremely expensive and rare. Gutenberg's innovation dramatically reduced the cost of producing written materials, enabling wider literacy and the rapid dissemination of ideas. The printing revolution contributed directly to the Protestant Reformation, the Scientific Revolution, and the broader democratization of knowledge. By 1500, an estimated twenty million volumes had been printed across Europe. The parallels between Gutenberg's press and the modern internet as transformative communication technologies are frequently noted by historians.",

        "Glaciology, the study of glaciers and ice sheets, provides critical insights into Earth's climate history. Ice cores drilled from Antarctic and Greenland ice sheets contain trapped air bubbles that preserve atmospheric composition dating back hundreds of thousands of years. Analysis of oxygen isotope ratios in these cores reveals detailed temperature records, while volcanic ash layers and dust concentrations indicate major geological events. The current rate of glacial retreat worldwide serves as one of the most visible indicators of ongoing climate change. Mountain glaciers in the Alps, Andes, and Himalayas have lost significant mass since the mid-nineteenth century, affecting water supplies for billions of people downstream.",

        "The culinary traditions of Japan emphasize seasonal ingredients, precise technique, and aesthetic presentation. Washoku, traditional Japanese cuisine, was inscribed on UNESCO's Intangible Cultural Heritage list in 2013. Key principles include using five colors, five cooking methods, and five flavors in each meal to achieve nutritional balance and visual harmony. Fermented foods like miso, soy sauce, and pickled vegetables play a central role, providing complex umami flavors that define the cuisine. The concept of shun refers to eating ingredients at their peak seasonal availability, a practice that connects diners to the natural rhythms of the agricultural calendar and local geography.",

        "The field of cryptography has evolved from simple substitution ciphers used by Julius Caesar to the complex mathematical algorithms that secure modern digital communications. The Enigma machine used by Nazi Germany during World War II represented a sophisticated mechanical encryption device that was famously broken by Allied codebreakers at Bletchley Park. Modern encryption relies on the computational difficulty of mathematical problems such as integer factorization and discrete logarithms. Public-key cryptography, developed in the 1970s, enables secure communication between parties who have never previously exchanged secret keys. Quantum computing poses both a threat to current encryption methods and an opportunity for theoretically unbreakable quantum key distribution.",

        "The architecture of ancient Rome continues to influence building design two thousand years later. Roman engineers developed concrete that could set underwater, enabling the construction of harbors, aqueducts, and the iconic dome of the Pantheon. The arch and vault systems they perfected distributed weight efficiently, allowing the creation of massive enclosed spaces like the Colosseum, which could seat fifty thousand spectators. Roman road-building techniques connected an empire spanning three continents, with some original roads still visible today. The principles of Roman urban planning, including grid street layouts, public forums, and centralized water systems, were rediscovered during the Renaissance and continue to inform modern city design.",
    ]

    def build_niah_prompt(target_tokens, needle_position_pct):
        """Build a long context prompt with a needle at the specified position."""
        # Rough estimate: 1 token ≈ 4 chars (conservative for English)
        target_chars = target_tokens * 4
        needle_chars = len(NEEDLE)
        question_chars = len(NEEDLE_QUESTION)
        filler_chars_needed = target_chars - needle_chars - question_chars - 200  # margin

        # Build filler by cycling through paragraphs
        filler_parts = []
        total = 0
        idx = 0
        while total < filler_chars_needed:
            para = FILLER[idx % len(FILLER)]
            filler_parts.append(para)
            total += len(para) + 2  # +2 for newlines
            idx += 1

        filler_text = "\n\n".join(filler_parts)

        # Trim to exact target
        if len(filler_text) > filler_chars_needed:
            filler_text = filler_text[:filler_chars_needed]

        # Split at needle position and inject
        insert_at = int(len(filler_text) * needle_position_pct)
        # Find next paragraph boundary to keep clean
        boundary = filler_text.find("\n\n", insert_at)
        if boundary == -1 or boundary > insert_at + 500:
            boundary = insert_at

        prompt = (
            "Read the following text carefully. You will be asked a question about it.\n\n"
            + filler_text[:boundary]
            + "\n\n" + NEEDLE + "\n\n"
            + filler_text[boundary:]
            + NEEDLE_QUESTION
        )
        return prompt

    # Define NIAH test configs based on model capabilities
    niah_configs = [
        ("niah_4k_mid", 4000, 0.5),
        ("niah_16k_mid", 16000, 0.5),
    ]
    if not is_mistral and not is_nano:
        niah_configs.extend([
            ("niah_32k_early", 32000, 0.1),
            ("niah_32k_mid", 32000, 0.5),
            ("niah_32k_late", 32000, 0.9),
            ("niah_64k_mid", 64000, 0.5),
        ])

    for name, target_tokens, position in niah_configs:
        print(f"  NIAH: {name} ({target_tokens//1000}K, pos={position:.0%})...", file=sys.stderr)
        prompt = build_niah_prompt(target_tokens, position)
        # Long contexts need longer timeouts: ~1s per 1K tokens for prefill + decode
        timeout_s = max(120, target_tokens // 500)
        r = test(name, prompt, max_tokens=50, pattern=NEEDLE_PATTERN, timeout_s=timeout_s)
        results.append(r)
        status = "PASS" if r["status"] == "PASS" else "FAIL"
        ttft = r.get("ttft_ms", 0)
        ptok = r.get("prompt_tokens", 0)
        print(f"    {status} (prompt={ptok} tok, TTFT={ttft:.0f}ms)", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# Output results
# ═══════════════════════════════════════════════════════════════
out = {"model": MODEL, "results": results}
passes = sum(1 for r in results if r["status"] == "PASS")
total = len(results)
avg_toks = sum(r.get("tok_s", 0) for r in results) / max(total, 1)
avg_ttft = sum(r.get("ttft_ms", 0) for r in results) / max(total, 1)

# Separate standard vs NIAH summaries
std_results = [r for r in results if not r["test"].startswith("niah_")]
niah_results = [r for r in results if r["test"].startswith("niah_")]
std_pass = sum(1 for r in std_results if r["status"] == "PASS")
niah_pass = sum(1 for r in niah_results if r["status"] == "PASS")

out["summary"] = {
    "pass": passes,
    "total": total,
    "std_pass": std_pass,
    "std_total": len(std_results),
    "niah_pass": niah_pass,
    "niah_total": len(niah_results),
    "avg_tok_s": round(avg_toks, 1),
    "avg_ttft_ms": round(avg_ttft, 1),
}
print(json.dumps(out))
cleanup()
