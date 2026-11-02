import sys,os,time,subprocess,struct
sys.path.insert(0,os.path.expanduser('~/strata-mac-vision/serve_pkg'))
import serve.server as S
home=os.path.expanduser('~')
cfg={'exe':home+'/strata-mac-vision/strata-vision-metal','mmproj':home+'/models/qwen38fn/mmproj-Qwen3.8-Flash-Next-BF16.gguf','model':home+'/models/qwen38fn/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf','gpu':True,'max_tokens':1024,'idle_unload_s':6}
def n_enc():
    o=subprocess.run(['pgrep','-f','strata-vision-metal'],capture_output=True,text=True).stdout.split()
    return len([x for x in o if x!=str(os.getpid())])
v=S.Vision(cfg,log=None)
print('after init: proc',v.proc,'alive()',v.alive(),'idle_unloaded',v.idle_unloaded,'procs',n_enc(),flush=True)
data=open(home+'/strata-mac-vision/test-1.jpeg','rb').read()
t=time.time(); p,n=v.encode(data); print('encode 1 (cold start incl.) %.2fs tokens %d'%(time.time()-t,n),'procs',n_enc(),flush=True)
h=struct.unpack('<5i',open(p,'rb').read(20)); print('hdr',[hex(h[0])]+list(h[1:]),flush=True)
t=time.time(); v.encode(data); print('encode 2 (cache hit) %.4fs procs'%(time.time()-t),n_enc(),flush=True)
time.sleep(16); print('after idle: proc',v.proc,'alive()',v.alive(),'idle_unloaded',v.idle_unloaded,'procs',n_enc(),flush=True)
v.cache.clear()
t=time.time(); p,n=v.encode(data); print('encode 3 (restart after idle) %.2fs tokens %d'%(time.time()-t,n),'procs',n_enc(),flush=True)
v.shutdown(); print('shutdown procs',n_enc())
