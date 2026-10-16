import json,sys
# mkcfg.py SRC DST key=val ... ; env:KEY=VAL sets env; arg:--flag=VAL replaces arg value
src,dst=sys.argv[1:3]; c=json.load(open(src))
for kv in sys.argv[3:]:
    k,v=kv.split('=',1)
    if k.startswith('env:'):
        if v=='': c['env'].pop(k[4:],None)
        else: c['env'][k[4:]]=v
    elif k.startswith('arg:'):
        a=c['args']; i=a.index(k[4:]); a[i+1]=v
    elif k=='name': c['model_name']=v
json.dump(c,open(dst,'w'),indent=1)
