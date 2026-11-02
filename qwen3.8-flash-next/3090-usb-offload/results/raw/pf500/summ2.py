import re,sys,collections,statistics as st
f=sys.argv[1]
rx=re.compile(r"pf-mac: L(\d+) c(\d+) T=(\d+) cold=(\d+) pairs=(\d+) rows=(\d+) \| front ([\d.]+) \| x_fetch ([\d.]+) \| conv\+pairs ([\d.]+) \| blobs\(join\) ([\d.]+) \| metal\((\w+)\) ([\d.]+) \| y_conv ([\d.]+) \| y_push ([\d.]+) \| issue ([\d.]+) gpuwait ([\d.]+) submit ([\d.]+) period ([\d.]+)")
prompts=[];cur=[]
for ln in open(f):
    m=rx.search(ln)
    if m:
        g=m.groups(); cur.append(dict(L=int(g[0]),c=int(g[1]),T=int(g[2]),cold=int(g[3]),pairs=int(g[4]),rows=int(g[5]),front=float(g[6]),x=float(g[7]),conv=float(g[8]),join=float(g[9]),metal=float(g[11]),yconv=float(g[12]),ypush=float(g[13]),issue=float(g[14]),gpuw=float(g[15]),sub=float(g[16]),period=float(g[17])))
    elif 'strata serve: prompt' in ln and 'read in' in ln:
        prompts.append((ln.strip()[:110],cur)); cur=[]
for hdr,rows in prompts:
    print(hdr, len(rows))
    by=collections.defaultdict(list)
    for r in rows:
        if r['L']>=3: by[(r['T'],r['c'])].append(r)
    for k,v in sorted(by.items()):
        a=lambda key: st.mean(x[key] for x in v)
        print("  T=%d c%d n=%d rows/T=%.3f | front %.1f (issue %.1f gpuwait %.1f) x %.1f conv %.1f join %.1f sub %.1f metal %.1f yconv %.1f ypush %.1f | period %.1f"%(k[0],k[1],len(v),a('rows')/k[0],a('front'),a('issue'),a('gpuw'),a('x'),a('conv'),a('join'),a('sub'),a('metal'),a('yconv'),a('ypush'),a('period')))
    first=[r for r in rows if r['L']<3]
    print("  first layers: join", [round(r['join']) for r in first][:12], "front",[round(r['front']) for r in first][:12])
