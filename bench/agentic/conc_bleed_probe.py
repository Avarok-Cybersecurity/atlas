"""Does one sequence's content leak into another's output?

The identical-prompt probe CANNOT see this: if all N sequences carry the same
data, bleeding between them is invisible. So use prompts whose correct answers
share no vocabulary, and look for each prompt's marker words showing up in a
DIFFERENT prompt's answer.

Each prompt is also given a unique nonsense token in its text; a nonce
appearing in the wrong answer is unambiguous cross-sequence contamination,
not a plausible-continuation coincidence.
"""
import json
import sys
import threading
import os
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else '8888'
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODEL = os.environ.get('PROBE_MODEL', 'longcat-full')
URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'

# (nonce, prompt, words that belong ONLY to this stream)
STREAMS = [
    ("ZQFX", "Remember the code ZQFX. Name three primary colours, one per line.",
     ["zqfx", "red", "blue", "yellow"]),
    ("VBNK", "Remember the code VBNK. Name three countries in South America, one per line.",
     ["vbnk", "brazil", "peru", "chile", "argentina"]),
    ("MWJT", "Remember the code MWJT. Name three musical instruments, one per line.",
     ["mwjt", "piano", "guitar", "violin", "drum"]),
    ("HRPD", "Remember the code HRPD. Name three planets, one per line.",
     ["hrpd", "mars", "venus", "jupiter", "saturn"]),
]


def ask(prompt, out, idx):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        out[idx] = d["choices"][0]["message"]["content"]
    except Exception as e:                       # noqa: BLE001
        out[idx] = f"<ERROR {e}>"


bleeds = 0
for r in range(ROUNDS):
    out = [None] * len(STREAMS)
    ts = [threading.Thread(target=ask, args=(s[1], out, i))
          for i, s in enumerate(STREAMS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f'--- round {r} ---')
    for i, (nonce, _, _) in enumerate(STREAMS):
        txt = (out[i] or '').lower()
        foreign = [
            STREAMS[j][0] for j in range(len(STREAMS))
            if j != i and STREAMS[j][0].lower() in txt
        ]
        tag = f'  BLEED<-{foreign}' if foreign else ''
        if foreign:
            bleeds += 1
        print(f'  [{nonce}] {(out[i] or "")[:100]!r}{tag}')

print(f'\nresponses containing ANOTHER stream\'s nonce: {bleeds}')
print('(any nonzero = cross-sequence contamination)')
