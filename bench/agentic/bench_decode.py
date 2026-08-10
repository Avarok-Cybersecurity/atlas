import os, json,urllib.request,time,statistics as st,sys
model=sys.argv[1]; tag=sys.argv[2]
rates=[]
for i in range(3):
    b={'model':model,'messages':[{'role':'user','content':f'[{tag}{i}] Write a detailed technical explanation of paged KV cache eviction.'}],
       'max_tokens':400,'temperature':0.6,'ignore_eos':True,'chat_template_kwargs':{'enable_thinking':False}}
    r=urllib.request.Request(os.environ.get('ATLAS_URL','http://localhost:8888/v1/chat/completions'),data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    t0=time.time(); d=json.load(urllib.request.urlopen(r,timeout=600)); el=time.time()-t0
    rates.append(d['usage']['completion_tokens']/el)
print(f'  {tag}: {st.median(rates):.1f} tok/s  (runs: {" ".join(f"{x:.1f}" for x in rates)})')
