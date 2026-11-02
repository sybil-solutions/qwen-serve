import subprocess,sys,time,os
imgs=sys.argv[2].split(','); extra=sys.argv[3:]
outdir=os.path.expanduser('~/strata-mac-vision/out'); os.makedirs(outdir,exist_ok=True)
def wired():
    o=subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
    d={}
    for l in o.splitlines():
        if ':' in l: k,v=l.split(':',1); d[k.strip()]=v.strip().strip('.')
    pg=16384
    return 'wired %.2f GB free %.2f GB'%(int(d['Pages wired down'])*pg/1e9, int(d['Pages free'])*pg/1e9)
exe=os.path.expanduser('~/strata-mac-vision/strata-vision-metal')
args=[exe,'--mmproj',os.path.expanduser('~/models/qwen38fn/mmproj-Qwen3.8-Flash-Next-BF16.gguf'),'--model',os.path.expanduser('~/models/qwen38fn/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'),'--gpu']+extra
print('before:',wired(),flush=True)
t0=time.time()
p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(outdir+'/m2.stderr','a'),text=True,bufsize=1)
print(p.stdout.readline().strip(),'start %.1fs'%(time.time()-t0),wired(),flush=True)
def fp():
    r=subprocess.run(['footprint','-p',str(p.pid)],capture_output=True,text=True).stdout
    return [l.strip() for l in r.splitlines() if 'Footprint' in l or 'footprint' in l][:2]
print('footprint',fp(),flush=True)
for im in imgs:
    for i in range(2):
        out=outdir+'/m2_%s_%d.bin'%(os.path.basename(im),i)
        t=time.time(); p.stdin.write('ENC %s %s\n'%(os.path.expanduser(im),out)); p.stdin.flush()
        print(os.path.basename(im),i,p.stdout.readline().strip(),'wall %.0f ms'%((time.time()-t)*1000),flush=True)
print('after enc:',wired(),'footprint',fp(),flush=True)
p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait(15)
time.sleep(1); print('after quit:',wired(),flush=True)
