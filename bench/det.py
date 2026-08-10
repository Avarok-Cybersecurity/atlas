import json,urllib.request,collections,sys
MODEL=sys.argv[1] if len(sys.argv)>1 else "puzzle"
N=int(sys.argv[2]) if len(sys.argv)>2 else 10
def raw(p,mt=24):
    b={"model":MODEL,"prompt":p,"max_tokens":mt,"temperature":0.0,"seed":1234}
    r=urllib.request.Request("http://localhost:8888/v1/completions",data=json.dumps(b).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=900) as x: d=json.loads(x.read())
    return (d["choices"][0].get("text") or "")
P=("The tollhouse ledger recorded exactly seventeen barrels of tar on the fourth of March."
   "\n\nQuestion: How many barrels of tar did the tollhouse ledger record?\nAnswer:")
outs=[raw(P) for _ in range(N)]
c=collections.Counter(outs)
print(f"  DISTINCT {len(c)}/{N}")
for o,k in c.most_common(): print(f"    x{k}: {o[:60]!r}")
