| lever (cumulative) | model tok/s c1 (MTP, 1.95 acc/window) | ms/window | VRAM hit | cold experts (RAM/page cache) per token | measured (3 prompts, tok/s) |
|---|---|---|---|---|---|
| baseline = F1: 17 whole-resident layers, 8770 slots, 1.6 GiB locked RAM, 3.5 ms handshake, 0.25 ms/expert | 5.68 | 343.34 | 0.393 | 189.5 | 5.1 avg (4.4 / 5.6 / 5.4) |
| + slots 9108 (vram reserve 700->100 MiB) and RAM tier 4.3 GiB locked (headroom 4->1.5 GiB), 17 whole layers | 6.04 | 323.09 | 0.464 | 177.5 | not run servable; E4 with prefill-borrow (9964 slots, 19 layers) 5.4 avg - borrow crashes on a 2.5K prefill |
| + frequency-ranked placement (every layer rings, 3.2-3.5 ms each; the hottest 9108 pairs resident) | 6.8 | 286.91 | 0.741 | 133.6 | F5 (servable): 5.9 avg (4.9 / 7.4 / 5.5); E5 with borrow 6.1 avg |
| + RAM tier doubled again (12 GB for experts; not reachable on 16 GB with the engine resident) | 6.81 | 286.22 | 0.741 | 93.0 | not run |
| + Metal lane next to NEON (compute share 0.155->0.089 ms: 0.25 -> 0.184 effective) | 7.43 | 262.48 | 0.741 | 93.0 | not built |
| + handshake hidden behind the previous layer (3.5 -> 0.5 ms/layer) | 12.42 | 157.05 | 0.741 | 93.0 | not built (3.5 ms is the measured floor of the current protocol; USB DMA floor 1 ms) |
| + 4000 more VRAM slots (no 200K KV / vision, or a 2nd card) | 17.68 | 110.3 | 0.825 | 47.0 | not run |
