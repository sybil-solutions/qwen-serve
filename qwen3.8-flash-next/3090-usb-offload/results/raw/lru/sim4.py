import numpy as np, collections
from collections import OrderedDict
a=np.fromfile('$HOME/strata-tgcu-shim/runs/decode/lru_trace.bin',dtype=np.int32)
calls=[];i=0
while i+3<=len(a):
    layer,nt,nd=a[i:i+3]; i+=3
    if i+nd>len(a): break
    v=a[i:i+nd]; i+=nd
    calls.append((int(layer),(v&0xffff).tolist(),(v>>16).tolist()))
wins=[];cur=[];prev=-1
for c in calls:
    if c[0]<=prev: wins.append(cur);cur=[]
    cur.append(c);prev=c[0]
wins.append(cur)
bounds=[0,95,215,380,475,600,len(wins)]
prompt_of=np.zeros(len(wins),int)
for p in range(6): prompt_of[bounds[p]:bounds[p+1]]=p%3
L=48;E=512
# LRU sim to label misses (S=3870)
S=3870
c=OrderedDict(); miss={}  # (w,l) -> set of missed experts
for wi,w in enumerate(wins):
    for l,ex,s in w:
        call=[l*512+e for e in ex]
        ms=set()
        for k in call:
            if k in c: c.move_to_end(k)
        for k in call:
            if k not in c:
                ms.add(k%512)
                if len(c)>=S:
                    for kk in c:
                        if kk not in call: del c[kk];break
                c[k]=1
            c.move_to_end(k)
        miss[(wi,l)]=ms
# per window per layer cold sets
cold={}
for wi,w in enumerate(wins):
    for l,ex,s in w: cold[(wi,l)]=ex
tot_miss=sum(len(m) for m in miss.values())
print("total misses",tot_miss)
def evaluate(held, ratio, mode):
    # train co-occurrence C[l] (E x E) on windows whose prompt != held ; test on windows of prompt held
    C=np.zeros((L-1,E,E),dtype=np.float32)
    F=np.zeros((L,E),dtype=np.float32)
    for wi in range(len(wins)):
        if prompt_of[wi]==held: continue
        for l in range(L):
            ex=cold.get((wi,l))
            if not ex: continue
            F[l,ex]+=1
            if l<L-1:
                nx=cold.get((wi,l+1))
                if nx:
                    for e in ex: C[l,e,nx]+=1
    cov_num=0; cov_den=0; vol=0
    rs=C.sum(2,keepdims=True)+1e-6
    Cn=C/rs
    for wi in range(len(wins)):
        if prompt_of[wi]!=held: continue
        for l in range(L-1):
            m=miss.get((wi,l+1))
            if not m: continue
            ex=cold.get((wi,l))
            if mode=="co" and ex: sc=Cn[l,ex].sum(0)
            elif mode=="freq": sc=F[l+1].copy()
            else: sc=np.zeros(E)
            n=int(np.ceil(ratio*len(m)))
            top=np.argpartition(-sc,n)[:n]
            cov_num+=len(m&set(top.tolist())); cov_den+=len(m); vol+=n
    return cov_num/cov_den, vol/cov_den
for held in (0,1,2):
    for mode in ("freq","co"):
        for ratio in (1.0,1.3,2.0):
            cv,vo=evaluate(held,ratio,mode)
            print(f"held-out p{held+1} {mode} ratio {ratio}: coverage {100*cv:.1f}% read volume x{vo:.2f}")
