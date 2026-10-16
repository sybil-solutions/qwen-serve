import sys, collections
import numpy as np
from collections import OrderedDict
a=np.fromfile('lru_trace.bin',dtype=np.int32)
calls=[];i=0
while i+3<=len(a):
    layer,nt,nd=a[i:i+3]; i+=3
    if i+nd>len(a): break
    v=a[i:i+nd]; i+=nd
    calls.append((int(layer),(v&0xffff).tolist(),(v>>16).tolist()))
print("calls",len(calls))
full=[[l*512+e for e in ex] for l,ex,s in calls]
tierset=set(l*512+e for l,ex,s in calls for e,st in zip(ex,s) if st==0)
print("tier set size seen",len(tierset), "distinct cold keys", len({k for c in full for k in c}))
def lru(stream,S):
    c=OrderedDict();h=m=0
    for call in stream:
        for k in call:
            if k in c: c.move_to_end(k); h+=1
        for k in call:
            if k not in c:
                m+=1
                if len(c)>=S:
                    for kk in c:
                        if kk not in call: del c[kk];break
                c[k]=1
            c.move_to_end(k)
    return h,m
tot=sum(len(c) for c in full)
for S in (1000,2000,3000,3500,3870,4500,5000):
    h,m=lru(full,S); print(f"merged LRU S={S}: hit {100*h/(h+m):.1f}%  miss/call {m/len(calls):.2f}")
nt=[[k for k in c if k not in tierset] for c in full]
tt=sum(len(c) for c in nt)
for S in (1000,1500,2000):
    h,m=lru(nt,S); inplace=(tot-tt+h)/tot
    print(f"tier(static {len(tierset)}) + LRU {S}: in-place {100*inplace:.1f}%  miss/call {m/len(calls):.2f}")
print("---- splits at total 3870 slots, tier = top-frequency keys of the trace (profile proxy)")
cnt=collections.Counter(k for c in full for k in c)
rank=[k for k,_ in cnt.most_common()]
for T in (0,500,1000,1500,2000,2876):
    ts=set(rank[:T]); st=[[k for k in c if k not in ts] for c in full]
    th=sum(1 for c in full for k in c if k in ts)
    h,m=lru(st,3870-T) if 3870-T>0 else (0,sum(len(c) for c in st))
    print(f"tier {T} + LRU {3870-T}: in-place {100*(th+h)/tot:.1f}% miss/call {m/len(calls):.2f}")
