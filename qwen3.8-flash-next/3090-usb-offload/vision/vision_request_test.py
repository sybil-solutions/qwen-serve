#!/usr/bin/env python3
"""Image request through the server: python3 vision_request_test.py [image] [url]  (expects the NYT 'MEN WALK ON MOON' photo)."""
import base64, json, sys, time, urllib.request, os
img = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/strata-mac-vision/test-1.jpeg")
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080/v1/chat/completions"
b64 = base64.b64encode(open(img, "rb").read()).decode()
mime = "image/png" if img.endswith(".png") else "image/jpeg"
body = {"model": "x", "max_tokens": 80, "temperature": 0, "stream": False,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": "What newspaper is this and what is the headline? Answer in one sentence."}]}]}
t = time.time()
r = urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600)
d = json.load(r)
print("wall %.1f s" % (time.time() - t)); print(json.dumps(d.get("usage"))); print(d["choices"][0]["message"].get("content"))
