#!/bin/bash
# Long-context coherence test for Atlas.
# Tests fibonacci generation with realistic agentic system prompts.
# Usage: ./test_long_context.sh [host:port] [model_hf_id]

set -euo pipefail

HOST="${1:-localhost:8888}"
MODEL="${2:-$(curl -s http://$HOST/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")}"
FIXTURES="$(dirname "$0")/fixtures"

echo "═══════════════════════════════════════════════"
echo "  Long-Context Coherence Test"
echo "  Model: $MODEL"
echo "  Host:  $HOST"
echo "═══════════════════════════════════════════════"

PASS=0
FAIL=0

run_fib_test() {
    local NAME="$1"
    local SYSTEM_PROMPT="$2"

    local RESULT
    if [ -z "$SYSTEM_PROMPT" ]; then
        RESULT=$(curl -s --max-time 120 "http://$HOST/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function called fibonacci(n) returning the nth fib number, 0-indexed (fibonacci(0)=0, fibonacci(1)=1). ONLY code.\"}],\"max_tokens\":300,\"temperature\":0}")
    else
        # Escape the system prompt for JSON
        local ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$SYSTEM_PROMPT")
        RESULT=$(curl -s --max-time 120 "http://$HOST/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"system\",\"content\":$ESCAPED},{\"role\":\"user\",\"content\":\"Write a Python function called fibonacci(n) returning the nth fib number, 0-indexed (fibonacci(0)=0, fibonacci(1)=1). ONLY code.\"}],\"max_tokens\":300,\"temperature\":0}")
    fi

    local FIB=$(echo "$RESULT" | python3 -c "
import json,sys,re
r=json.load(sys.stdin)
if 'error' in r:
    print('ERROR: ' + str(r.get('error',{}).get('message',r['error'])))
    sys.exit(0)
c=r['choices'][0]['message']['content']
m=re.search(r'\x60\x60\x60python\s*\n(.*?)\x60\x60\x60',c,re.DOTALL)
code=m.group(1) if m else c.strip()
try:
    ns={}; exec(code,ns)
    fn=ns.get('fibonacci') or ns.get('fib')
    if fn is None:
        for v in ns.values():
            if callable(v): fn=v; break
    r=[fn(i) for i in range(10)]
    expected=[0,1,1,2,3,5,8,13,21,34]
    if r==expected: print('PASS')
    else: print('FAIL: got '+str(r))
except Exception as e:
    print('FAIL: '+str(e))
" 2>&1)

    local TOKENS=$(echo "$RESULT" | python3 -c "import json,sys;r=json.load(sys.stdin);print(r.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null)

    if [[ "$FIB" == "PASS" ]]; then
        echo "  ✓ $NAME ($TOKENS prompt tokens): PASS"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $NAME ($TOKENS prompt tokens): $FIB"
        FAIL=$((FAIL + 1))
    fi
}

# Test 1: Baseline (no system prompt, ~20 tokens)
run_fib_test "Baseline (no system prompt)" ""

# Test 2: Claude Code system prompt (~30K tokens)
CLAUDE_SP=$(cat "$FIXTURES/claude_code_system_prompt.txt")
run_fib_test "Claude Code system prompt (~30K tok)" "$CLAUDE_SP"

# Test 3: OpenCode system prompt (~7K tokens)
OPENCODE_SP=$(cat "$FIXTURES/opencode_system_prompt.txt")
run_fib_test "OpenCode system prompt (~7K tok)" "$OPENCODE_SP"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════════"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
