import struct, sys, collections, heapq
import numpy as np
raw=open(sys.argv[1],'rb').read()
a=np.frombuffer(raw,dtype=np.int32)
calls=[]; i=0
while i+3<=len(a):
    layer,nt,nd=a[i:i+3]; i+=3
    if i+nd>len(a): break
    v=a[i:i+nd]; i+=nd
    calls.append((int(layer),int(nt),(v&0xffff).tolist(),(v>>16).tolist()))
print("calls",len(calls))
st=collections.Counter()
for l,nt,ex,s in calls:
    for x in s: st[x]+=1
tot=sum(st.values())
print("status counts (0 tier,1 lru hit,2 miss cached,3 miss ssd):",{k:(v,round(v/tot*100,1)) for k,v in sorted(st.items())}, "per call", tot/len(calls))
# split into windows: new window when layer <= previous layer
wins=[]; cur=[]; prev=-1
for c in calls:
    if c[0]<=prev: wins.append(cur); cur=[]
    cur.append(c); prev=c[0]
wins.append(cur)
print("windows",len(wins))
# stream of (layer, expert) per call excluding tier (status 0)
stream=[[ (l*512+e) for e,s_ in zip(ex,s) if s_!=0] for l,nt,ex,s in calls]
S=int(sys.argv[2]) if len(sys.argv)>2 else 1000
def lru(S):
    from collections import OrderedDict
    c=OrderedDict(); h=m=0
    for call in stream:
        for k in call:
            if k in c: c.move_to_end(k); h+=1
        for k in call:
            if k not in c:
                m+=1
                if len(c)>=S:
                    # evict oldest not in this call
                    for kk in c:
                        if kk not in call: del c[kk]; break
                c[k]=1
            c.move_to_end(k)
    return h,m
def lfu(S,decay=0.0, win=0):
    cnt={}; h=m=0; slots=set()
    for call in stream:
        cs=set(call)
        for k in call:
            if k in slots: h+=1
        for k in call:
            cnt[k]=cnt.get(k,0)*1.0+1
        for k in call:
            if k not in slots:
                m+=1
                if len(slots)>=S:
                    v=min((s for s in slots if s not in cs), key=lambda s:cnt[s])
                    slots.discard(v)
                slots.add(k)
    return h,m
def slru(S,prot=0.5):
    from collections import OrderedDict
    pb=OrderedDict(); pr=OrderedDict(); P=int(S*prot); h=m=0
    for call in stream:
        cs=set(call)
        for k in call:
            if k in pr: pr.move_to_end(k); h+=1
            elif k in pb:
                h+=1; del pb[k]; pr[k]=1
                if len(pr)>P:
                    kk,_=pr.popitem(last=False); pb[kk]=1
        for k in call:
            if k not in pr and k not in pb:
                m+=1
                while len(pr)+len(pb)>=S:
                    for kk in pb:
                        if kk not in cs: del pb[kk]; break
                    else:
                        for kk in pr:
                            if kk not in cs: del pr[kk]; break
                        else: break
                pb[k]=1
    return h,m
def belady(S):
    # next-use oracle
    nxt={}; nu=[]
    flat=[]
    for ci,call in enumerate(stream):
        for k in call: flat.append((ci,k))
    nextuse=[0]*len(flat); last={}
    for idx in range(len(flat)-1,-1,-1):
        ci,k=flat[idx]; nextuse[idx]=last.get(k,1<<60); last[k]=idx
    slots={}; h=m=0; idx=0
    for ci,call in enumerate(stream):
        cs=set(call)
        base=idx
        for j,k in enumerate(call):
            if k in slots: h+=1; slots[k]=nextuse[base+j]
        for j,k in enumerate(call):
            if k not in slots:
                m+=1
                if len(slots)>=S:
                    v=max((s for s in slots if s not in cs), key=lambda s:slots[s]); del slots[v]
                slots[k]=nextuse[base+j]
        idx+=len(call)
    return h,m
for S in (1000,1500,2000,3000):
    for name,f in (("lru",lru),("lfu",lfu),("slru.5",slru),("belady",belady)):
        h,m=f(S); print(f"S={S} {name}: hits {h} misses {m} hit {100*h/(h+m):.1f}%  misses/call {m/len(calls):.2f}")
