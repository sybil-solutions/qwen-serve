import numpy as np, collections
from collections import OrderedDict
a=np.fromfile('lru_trace.bin',dtype=np.int32)
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
print("windows",len(wins))
# per-window cold entries count
ent=[sum(len(c[1]) for c in w) for w in wins]
print("entries per window quartiles", np.percentile(ent,[10,50,90]))
# detect prompt boundaries: windows with big entries (p1 ~ 23*48=1100) vs p2 ~ 500
import sys
def run(S,tierset=None):
    c=OrderedDict(); res=[]
    for w in wins:
        h=m=0
        for l,ex,s in w:
            call=[l*512+e for e in ex]
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
        res.append((h,m,len(w)))
    return res
for S in (1000,3870):
    r=run(S)
    print("S",S)
    # print block stats of window-miss counts in chunks of 40 windows
    for b in range(0,len(r),40):
        blk=r[b:b+40]; h=sum(x[0] for x in blk); m=sum(x[1] for x in blk)
        print(f" win {b:4d}-{b+len(blk):4d}: hit {100*h/(h+m):5.1f}% miss/window {m/len(blk):6.1f}  entries/window {(h+m)/len(blk):6.1f}")
print("==== larger S: miss/window for p1a(0-90) p2a(90-210) p3a(210-360) p1b(400-490) p2b(490-610) p3b(610-766)")
for S in (3870,4500,5000,5500,6500,8000,10000):
    r=run(S)
    def mw(a,b):
        blk=r[a:b]; return sum(x[1] for x in blk)/len(blk)
    print(S, [round(mw(a,b),1) for a,b in ((0,90),(90,210),(210,360),(400,490),(490,610),(610,766))])
