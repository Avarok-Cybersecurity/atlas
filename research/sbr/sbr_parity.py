#!/usr/bin/env python3
"""SBR M1 exactness parity — tail-pin vs baseline (full-replay reference).

Tail-pin changes only WHICH checkpoint is restored; replay from it is the same
bit-exact WY4 path, so greedy (temp=0) generations MUST be token-identical to the
baseline that replays from a deeper/closer checkpoint. This harness runs an
identical set of N warm-resume prompts on whatever server is up, recording the
greedy continuation + per-token top-logprob. Run once per config (baseline,
tail-pin), then `--compare a.json b.json` for argmax-agreement + KL.

  python3 sbr_parity.py --label baseline --out p_base.json --n 10
  python3 sbr_parity.py --label tailpin  --out p_pin.json  --n 10
  python3 sbr_parity.py --compare p_base.json p_pin.json
"""
import argparse, json, math, time, urllib.request

WORDS=("the recurrent state integrates each token through a gated delta update while "
       "attention reads cached keys and values across the long context window").split()
def filler(n,s):
    m=int(n/0.75); return f"[doc {s}] "+" ".join(WORDS[(i+s)%len(WORDS)] for i in range(m))

def gen(base_url, model, messages, max_tokens=64):
    body=json.dumps({"model":model,"messages":messages,"max_tokens":max_tokens,
                     "temperature":0.0,"logprobs":True,"top_logprobs":1,"stream":False}).encode()
    req=urllib.request.Request(base_url+"/chat/completions",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=900) as r: d=json.loads(r.read())
    ch=d["choices"][0]; text=ch["message"]["content"]
    toks=[]
    lp=ch.get("logprobs") or {}
    for c in lp.get("content",[]):
        toks.append({"tok":c.get("token"),"lp":c.get("logprob")})
    return text, toks

def run(a):
    # N deep multi-turn prompts (distinct), each a warm-resume-shaped prompt
    res=[]
    for i in range(a.n):
        msgs=[{"role":"system","content":"Analyze.\n"+filler(3000,i)},
              {"role":"user","content":f"Q{i}: {filler(1500,i+50)} Summarize in one paragraph."},
              {"role":"assistant","content":filler(400,i+100)},
              {"role":"user","content":"Given all the above, state the single key conclusion."}]
        text,toks=gen(a.base_url,a.model,msgs,64)
        res.append({"i":i,"text":text,"toks":toks})
        print(f"[{a.label}] prompt {i}: {text[:60]!r}",flush=True)
    json.dump({"label":a.label,"res":res},open(a.out,"w"),indent=2)

def compare(pa,pb):
    A=json.load(open(pa))["res"]; B=json.load(open(pb))["res"]
    assert len(A)==len(B)
    n_text_eq=0; tot_tok=0; agree=0; kl_sum=0.0; kl_n=0
    for a,b in zip(A,B):
        if a["text"]==b["text"]: n_text_eq+=1
        for ta,tb in zip(a["toks"],b["toks"]):
            tot_tok+=1
            if ta["tok"]==tb["tok"]: agree+=1
            if ta["lp"] is not None and tb["lp"] is not None:
                pa_=math.exp(ta["lp"]); pb_=math.exp(tb["lp"])
                if pa_>0 and pb_>0: kl_sum+=pa_*math.log(pa_/pb_); kl_n+=1
    print(f"text-identical: {n_text_eq}/{len(A)}")
    print(f"argmax agreement: {agree}/{tot_tok} = {100*agree/max(tot_tok,1):.2f}%")
    print(f"mean top-token KL: {kl_sum/max(kl_n,1):.3e} (n={kl_n})")
    print("VERDICT:", "EXACT (parity)" if n_text_eq==len(A) and agree==tot_tok else "DIVERGENCE")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--out"); ap.add_argument("--n",type=int,default=10)
    ap.add_argument("--base-url",default="http://localhost:8888/v1")
    ap.add_argument("--model",default="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    ap.add_argument("--compare",nargs=2)
    a=ap.parse_args()
    if a.compare: compare(*a.compare)
    else: run(a)
