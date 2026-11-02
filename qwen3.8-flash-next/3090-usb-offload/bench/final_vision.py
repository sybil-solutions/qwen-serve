#!/usr/bin/env python3
"""final_vision.py: image requests (stream) with TTFT; builds chart.png if missing. usage: final_vision.py [image question]..."""
import base64, json, sys, time, urllib.request, os
from PIL import Image, ImageDraw, ImageFont
def chart(p):
    im=Image.new("RGB",(900,600),"white"); d=ImageDraw.Draw(im)
    try: f=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",30); fs=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",24)
    except Exception: f=fs=ImageFont.load_default()
    d.text((200,20),"Quarterly revenue 2025 (USD millions)",fill="black",font=f)
    vals=[("Q1",12,"#3b82f6"),("Q2",30,"#10b981"),("Q3",21,"#f59e0b"),("Q4",45,"#ef4444")]
    d.line((100,540,860,540),fill="black",width=3); d.line((100,100,100,540),fill="black",width=3)
    for i,(q,v,c) in enumerate(vals):
        x=150+i*180; h=v*9; d.rectangle((x,540-h,x+110,540),fill=c); d.text((x+30,545),q,fill="black",font=fs); d.text((x+30,540-h-34),str(v),fill="black",font=fs)
    im.save(p)
HOME=os.path.expanduser("~/strata-mac-vision")
if not os.path.exists(HOME+"/chart.png"): chart(HOME+"/chart.png")
TESTS=[(HOME+"/test-1.jpeg","What newspaper is this and what is the headline? Answer in one sentence."),
       (HOME+"/chart.png","Which quarter has the highest revenue and what is its value? Which has the lowest? Answer briefly.")]
if len(sys.argv)>2: TESTS=[(sys.argv[1],sys.argv[2])]
for img,q in TESTS:
    b64=base64.b64encode(open(img,"rb").read()).decode(); mime="image/png" if img.endswith(".png") else "image/jpeg"
    body={"model":"x","max_tokens":120,"temperature":0,"stream":True,"reasoning_effort":"none","stream_options":{"include_usage":True},
          "messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}},{"type":"text","text":q}]}]}
    t=time.time(); ttft=None; out=""; usage=None; tm=None
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=3600)
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        j=json.loads(line[5:])
        if j.get("usage"): usage=j["usage"]
        if j.get("timings"): tm=j["timings"]
        for c in j.get("choices",[]):
            dl=c.get("delta",{}).get("content")
            if dl:
                if ttft is None: ttft=time.time()-t
                out+=dl
    print(json.dumps({"image":os.path.basename(img),"ttft_s":round(ttft or -1,2),"wall_s":round(time.time()-t,2),"usage":usage,"timings":tm}))
    print("ANSWER:",out.strip(),flush=True)
