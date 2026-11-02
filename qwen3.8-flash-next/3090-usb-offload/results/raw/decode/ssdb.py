import os, sys, time, random, threading, fcntl
F_NOCACHE=48
path=os.path.expanduser('~/exl3/pack_full512/experts.bin')
fd=os.open(path, os.O_RDONLY)
size=os.fstat(fd).st_size
B=1862400
nblob=size//B
def run(nthreads, chunks, nocache, nreads=96, seed=1):
    random.seed(seed)
    fcntl.fcntl(fd, F_NOCACHE, 1 if nocache else 0)
    offs=[random.randrange(0,nblob-1)*B for _ in range(nreads)]
    jobs=[]
    for o in offs:
        cb=(B//chunks+4095)&~4095
        p=0
        while p<B:
            jobs.append((o+p,min(cb,B-p))); p+=cb
    idx=[0]; lk=threading.Lock()
    def w():
        buf=bytearray(B)
        mv=memoryview(buf)
        while True:
            with lk:
                i=idx[0]; idx[0]+=1
            if i>=len(jobs): return
            o,n=jobs[i]
            got=0
            while got<n:
                r=os.preadv(fd,[mv[got:n]],o+got)
                if r<=0: break
                got+=r
    t=time.time()
    ths=[threading.Thread(target=w) for _ in range(nthreads)]
    [x.start() for x in ths]; [x.join() for x in ths]
    dt=time.time()-t
    return nreads*B/1e9/dt, dt/nreads*1000
for nocache in (0,1):
    for nt,ch in ((1,1),(4,1),(7,1),(7,4),(8,8),(16,4)):
        seed=random.randrange(1<<30)
        g,ms=run(nt,ch,nocache,seed=seed)
        print(f"nocache={nocache} threads={nt} chunks/blob={ch}: {g:.2f} GB/s, {ms:.2f} ms/blob",flush=True)
