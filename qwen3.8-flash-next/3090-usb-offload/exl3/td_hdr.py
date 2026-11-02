"""td_hdr.py: fetch the 7 safetensors headers of turboderp/Qwen3.8-Flash-Next-exl3@69e33439 into <this directory>/td/experts_index.json + other_index.json
(only the layer expert tensors + a few others), and check each expert's 12 tensors form ONE contiguous byte range."""
import json, os, re, struct, sys, time, urllib.request
REV = "69e33439ae950f17bcbe95c98f117d80f759ab6d"
BASE = f"https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/resolve/{REV}/"
def rng(shard, a, b, tries=5):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(BASE + shard, headers={"Range": f"bytes={a}-{b}"}), timeout=120).read()
        except Exception as e:
            if t == tries - 1: raise
            time.sleep(2 * (t + 1))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "td"); os.makedirs(out, exist_ok=True)
pat = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(trellis|suh|svh|mul1)$")
res, other = {}, {}
for i in range(1, 8):
    sh = f"model-0000{i}-of-00007.safetensors"
    n = struct.unpack("<Q", rng(sh, 0, 7))[0]
    h = json.loads(rng(sh, 8, 8 + n - 1))
    for k, v in h.items():
        if k == "__metadata__": continue
        m = pat.match(k)
        if m:
            l, e, q, f = int(m[1]), int(m[2]), m[3], m[4]
            res.setdefault(f"{l}.{e}", {})[f"{q}_{f}"] = [sh, 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1], v["dtype"], v["shape"]]
        else:
            other[k] = [sh, 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1], v["dtype"], v["shape"]]
    print(sh, "header", n, "experts so far", len(res), flush=True)
json.dump(res, open(f"{out}/experts_index.json", "w")); json.dump(other, open(f"{out}/other_index.json", "w"))
# contiguity per expert
bad = 0; spans = {}
for key, d in res.items():
    assert len(d) == 12, (key, len(d))
    sh = {v[0] for v in d.values()}
    iv = sorted((v[1], v[2]) for v in d.values())
    contiguous = len(sh) == 1 and all(iv[j][1] == iv[j + 1][0] for j in range(len(iv) - 1))
    if not contiguous: bad += 1
    spans[key] = (sh.pop() if len(sh) == 1 else None, iv[0][0], iv[-1][1])
print("experts", len(res), "non-contiguous", bad)
import collections
print("span sizes", collections.Counter(b - a for _, a, b in spans.values()))
shape_ct = collections.Counter((k.split("_")[0], k.split("_")[1], tuple(v[4]), v[3]) for d in res.values() for k, v in d.items())
for k, v in sorted(shape_ct.items()): print(k, v)
print("other keys", len(other)); 
for k in list(other)[:5]: print(k)
