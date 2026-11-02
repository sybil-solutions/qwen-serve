# Results

Every number here comes from a log or receipt in this repo; the path is in the "source" column. Times are UTC as written in the logs.
Nothing is rounded up. "Not run" means not run.

Rig: Mac mini M4 (Mac16,10, 10 cores, 16 GiB, internal NVMe) running macOS 26.6.2, driving one RTX 3090 24 GB over a "chestnut"
USB dock (ASM2464 bridge, GPU link Gen3 x2, 2026-10-10 `LNKSTA 0x1023`) through the tgcu CUDA shim on tinygrad's NV backend.
Engine: Strata v0.1.41 + the patches in [`../patches/`](../patches/README.md). Weights: turboderp EXL3 3.05 bpw, all 512 experts of all 48 MoE
layers (45.77 GB of experts), dense part IQ3_S GSQ-RCO GGUF, MTP draft layer from the BF16 checkpoint (see [reproduce.md](reproduce.md)).

Method (all rows unless a row says otherwise): one request at a time (concurrency 1), greedy (`temperature 0`), reasoning off, MTP
speculative decoding (`--spec 4 --spec-min-p 0.5`), `--kv k8v4 --max-context 204800`, vision encoder configured, all 512 experts computed.
Decode: 3 fixed prompts, up to 300 tokens, run twice right after a server restart (`bench/split_probe.py`); the tok/s is
`predicted_n / predicted_ms` from the server's `timings` (it counts every token the verify windows accepted, MTP drafts included).
Prefill: a document prompt with a random first line so no prefix cache applies, `max_tokens` 2, tok/s = `prompt_n / prompt_ms`
(`bench/ls_pf.py`, `bench/final_long.py pf`). Needle: `bench/final_long.py needle TOKENS DEPTH` (Strata's `tools/needle_bench.py` haystack).

## 0. Latest configuration: `configs/strata-exl3-full512-final-g4-lru-ht.json.tmpl` (host-tail step)

= the configuration of section 1 plus `TGCU_POKE1=15` (GPPut / doorbell / hold-word as one control transfer), `TGCU_FREE_DEFER=1` (cudaFree hands the block back when its submission completed instead of a
full synchronize per free), `TGCU_H2D_RELAX=1` (a bulk push waits only for the launches whose parameters fall inside its destination range), `TGCU_TS_G=150` (compute channel group time slice
1024 -> 150 us: the copy engine is no longer slowed by a busy GPU during the decode window), `STRATA_PF_SPLITF=1` with `STRATA_PF_SPLITF_MIN=4` (patch `0005`: the split order only for groups of 4+ chunks) and
`EXL3M_WAIT_SPIN=1` (decode Metal jobs poll the command buffer). The sources of these changes are in the repo (`shim/tgcu*.py` and `exl3/metal` carry them, `patches/strata/0005`, the per-step diffs in `patches/history/hosttail/`);
`libexl3metal.dylib` rebuilt from the repo has the same sha256 as the deployed one (`e38a011c79933a8e...`), the patched `shim/*.py` files and `prefill.cpp` are byte-identical to the deployed ones (checked 2026-10-12).
Run 2026-10-12 07:26:46Z to 07:34:39Z on a freshly restarted server, same weights/placement/memory layout as section 1 (7,400 VRAM slots, 716 MiB free, LRU 5,600 slots).
Source: [`results/raw/hosttail/ht_final.log`](../results/raw/hosttail/ht_final.log) (`split_HF_{a,b,c}_{0,1,2}.txt` and `split_HT9_st_*.txt` in the same directory are the generated texts of the three decode probes of the tuning/instrumented runs, not timings).

### 0.1 Decode, c1, MTP (tok/s)

| prompt | run 1 | run 2 | after the prefill rows + 32K needle | MTP drafts accepted/proposed (run 1) |
|---|---|---|---|---|
| 1: "write is_prime and explain it" (29 prompt tokens, 300 out) | **18.2** | 17.9 | 17.5 | 213/258 |
| 2: "explain how an MoE picks experts" (35 prompt tokens, 253 out) | **21.4** | **21.3** | 21.5 | 134/225 |
| 3: "lighthouse keeper story" (29 prompt tokens, 292 out) | **18.1** | 17.7 | 17.5 | 138/260 |

Time to first token for these 29-35 token prompts is 2.1-2.5 s. (Section 1 was 16.8 / 19.0 / 16.3 and 16.6 / 19.0 / 16.0.) The stats-instrumented rerun of the same config
([`ht9_st.log`](../results/raw/hosttail/ht9_st.log), `STRATA_PF_MAC_STATS` + `STRATA_PF_MAC_RDSTATS` on): 18.6 / 21.4 / 17.9.

### 0.2 Prefill and context, c1 (tok/s)

| prompt tokens | prefill speed | decode over the answer | note |
|---|---|---|---|
| 3,783 | 277.3, 333.6 | not measured | two runs, different random first line (276.2, 325.8 in `ht9_st.log`) |
| 7,905 | 427.8 (cold), 481.3 (warm) | not measured | 431.3, 482.1 in `ht9_st.log`; the cold-vs-warm gap (~2.1 s) is layers 0-1 waiting for SSD reads of experts that are not in the LRU |
| 17,610 | 406.8 | not measured | 9 chunks -> groups of 3, `STRATA_PF_SPLITF_MIN=4` leaves the split order off there |
| 33,253 (32K needle) | 396.0 (TTFT 84.0 s) | 17.3 tok/s (124 tokens, MTP 67/112) | needle `meadow-copper-504`, depth 50 %, found |
| 97,461 / 188,835 | 374.9 / 382.4 (earlier config) | 16.5 / 13.9 (earlier config) | **not re-run** on this config, see section 1.3 |

Text parity: 24 greedy tokens at 3.8K and 7.9K are byte-identical to the reference run with the host-tail env off ([`ht_ex2_ts150.log`](../results/raw/hosttail/ht_ex2_ts150.log)). Vision: the `final_vision.py` step of
`ht_final.log` aborted (`No module named 'PIL'`); the plan notes report the image answers (NYT photo, bar chart) as OK on this config, there is no raw log of that in this repo. Prefill >= 500 tok/s was **not reached** (best 481 warm).

## 1. Previous configuration: `configs/strata-exl3-full512-final-g4-lru.json.tmpl`

Run `final_lru` on 2026-10-12, 06:12:48Z to 06:21:32Z, started on a freshly restarted server. 7,400 expert slots in VRAM (12.84 GiB),
716 MiB of VRAM free after load, Metal LRU of 5,600 registered slots (10.46 GB), 0.2 GiB static RAM tier, prefill group G=4.
Source for 1.1-1.3: [`results/raw/lru/final_lru.log`](../results/raw/lru/final_lru.log).

### 1.1 Decode, c1, MTP (tok/s)

| prompt | run 1 | run 2 | after the prefill rows + 32K needle | MTP drafts accepted/proposed (run 1) |
|---|---|---|---|---|
| 1: "write is_prime and explain it" (29 prompt tokens, 300 out) | 16.8 | 16.6 | 16.3 | 213/258 |
| 2: "explain how an MoE picks experts" (35 prompt tokens, 253 out) | **19.0** | **19.0** | 18.8 | 134/225 |
| 3: "lighthouse keeper story" (29 prompt tokens, 292 out) | 16.3 | 16.0 | 16.3 | 138/260 |

Time to first token for these 29-35 token prompts is 2.2-2.6 s.

### 1.2 Prefill, c1 (tok/s)

| prompt tokens | prefill speed (two runs of the same shape, different random first line) | note |
|---|---|---|
| 3,783 | 276.9, 329.8 | first run after the decode probes, second run |
| 7,905 | 382.6, 421.7 | |
| 8,218 | 346.1 | one run |
| 17,610 | 392.1 | one run |
| 33,253 (32K needle) | 378.1 | TTFT 87.9 s, 32K needle found (`meadow-copper-504`, depth 50 %), decode 15.9 tok/s over 130 tokens |

### 1.3 Context

| prompt tokens | prefill speed | decode over the answer | needle | config | source |
|---|---|---|---|---|---|
| 33,253 | 378.1 tok/s (TTFT 87.9 s) | 15.9 tok/s (130 tokens, MTP 71/102) | found | final-g4-lru | `results/raw/lru/final_lru.log` |
| 97,461 | 374.9 tok/s (TTFT 260.0 s) | 16.5 tok/s (131 tokens, MTP 74/97) | found (depth 25 %) | `g4c` = final-g4 + `TGCU_HANG_DUMP=1 TGCU_WAIT_S=30` (diagnostic env only), before the LRU change | `results/raw/g4/lr_g4c_100k.log` |
| 188,835 | 382.4 tok/s (TTFT 493.8 s) | 13.9 tok/s (134 tokens, MTP 66/121) | found (depth 50 %) | final-g4 (before the LRU change) | `results/raw/g4/final_g4.log` |

The 200K context itself is allocated in every row above (`--max-context 204800`, k8v4 KV, 13 KV layers: 12 QSA + the MTP draft; the 36 GDN
layers carry constant state). The 97K and 189K rows were measured on the preceding config (LRU 1,000 slots, 5.1 GiB static tier): the 190K
needle was **not** re-run on the final LRU config.

### 1.4 Vision

An image request through `/v1/chat/completions` is answered correctly: the New York Times front page photo (headline "MEN WALK ON MOON",
640x488 and 1280x960) and a generated bar chart (Q4 = 45 highest, Q1 = 12 lowest). Encoder = `strata-vision-metal` on the mini (llama.cpp mtmd + the BF16
mmproj), 0 bytes of 3090 VRAM; 1.2 s to encode a 566-token chart, 2.5 s for 993 tokens, 4.1 s cold start + encode with `idle_unload_s: 120`.
Source: `results/campaign/GOAL.md` status table ("vision" row, 2026-10-11) and `results/campaign/EXL3_PLAN-log.md` 2026-10-12T00:05Z and 06:25Z.
**No raw vision log is in this repo**: `bench/final_vision.py` failed in the `final_lru` run (`No module named 'PIL'`, see the log); the answers on
the final-g4-lru config are reported in the 06:25Z plan entry, the encode timings are from the earlier final config.

## 2. Where the decode window goes (final vs the config it replaced)

`STRATA_DECODE_TIMING=1`, per prompt, ms per verify window; "transport" is derived: per-layer host - plan - jobs - CPU. 48 ring layers per window.
Sources: [`results/raw/lru/dd_e0.log`](../results/raw/lru/dd_e0.log) (live G=4 config before the change: LRU 1,000 slots + 5.1 GiB tier) and
[`dd_e7.log`](../results/raw/lru/dd_e7.log) (final: LRU 5,600 + `STRATA_MAC_PULL_NOCACHE=1`).

| prompt | config | tokens/window | ms/window | GPU-reach wait | cold-expert reads ("jobs") | Metal wait ("CPU") | transport | draft | tok/s (derived) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | before | 3.23 | 291.55 | 31.20 | 135.39 | 35.12 | 73.87 | 10.17 | 11.1 |
| 1 | final | 3.33 | 198.88 | 27.59 | 78.04 | 17.91 | 59.39 | 10.10 | 16.7 |
| 2 | before | 2.03 | 119.40 | 20.57 | 17.48 | 10.73 | 56.36 | 8.88 | 17.0 |
| 2 | final | 2.08 | 109.29 | 17.39 | 16.20 | 6.88 | 54.73 | 8.75 | 19.0 |
| 3 | before | 1.98 | 144.19 | 23.22 | 27.44 | 17.65 | 61.77 | 8.73 | 13.7 |
| 3 | final | 1.82 | 114.81 | 17.33 | 17.18 | 12.01 | 54.79 | 8.22 | 15.9 |

(The remaining ~6 ms of each window is stage + commit/emit.) Per ring layer, mean over 750 windows with `STRATA_HS_TRACE=1`
(plan entry 2026-10-12T06:25Z; the trace log itself is not in the repo): wait for the ring 0.431 ms, x pull residue 0.057, pool call 1.85
(SSD misses + Metal wait), y + flag push 0.419, flag 0.023, host after the ring 2.36, so about 2.8 ms x 48 layers = 134 ms of the window.
One pull and one push per ring is already the floor (two USB transactions = 0.85 ms).

The cold-expert placement per lookup (36.5K-call trace, 11.4 cold experts per call), before: static tier 22 %, LRU hit 45 %, miss 33 % (3.58 misses
per call, 67.5 % in place); after: 89.2 % in place, 1.23 misses per call (plan entry 06:25Z; trace `results/raw/decode/` has the sim scripts, not the 2 MB trace).

## 3. Decode progression, same three prompts (tok/s p1 / p2 / p3, c1, MTP)

Full-512 exact-experts configs only. Source for a row = the ledger row (`results/campaign/ledger.jsonl`) and the log named there, kept under `results/raw/`.

| date (UTC) | change | decode | source |
|---|---|---|---|
| 2026-10-11 17:45 | first end-to-end full-512 run, 17 whole layers resident, 1.6 GiB RAM tier | 4.4 / 5.6 / 5.4 | `mini-shim-logs/split_F1_full512_a.log` |
| 2026-10-11 18:20 | frequency-ranked placement, every layer rings | 5.2 / 7.4 / 5.7 | `mini-shim-logs/split_F4_full512_d_freq.log` |
| 2026-10-11 18:50 | + no prefill borrow (servable config) | 4.9 / 7.4 / 5.5 | `mini-shim-logs/split_F5_full512_e_freq_noborrow.log` |
| 2026-10-11 19:55 | pre-armed USB handshake + deferred pull + whole-blob preads (`HS_PRE`, `HS_DEFER`, `MAC_PULL`), texts byte-identical to the row above | 9.3 / 12.4 / 8.5 | `mini-shim-logs/split_H4_pull.log`, `split_H9_final.log` |
| 2026-10-11 21:00 | Metal lane with an LRU of 1,000 registered slots | 10.3 / 14.7 / 10.6 | `mini-shim-logs/split_WF6_m_final.log` |
| 2026-10-12 00:05 | + 200K k8v4 KV and vision (7,820 slots, -1,288) | 9.0 / 12.3 / 9.4 | ledger row 2026-10-12T00:05Z |
| 2026-10-12 02:07 | in-place/miss split job + early start (`METAL_SPLIT`, `METAL_EARLY`, `MAC_PULL_EARLY`) | 10.4 / 17.2 / 13.8 and 10.1 / 17.2 / 12.1 | `mini-shim-logs/dec_final.log` |
| 2026-10-12 05:28 | G=4 prefill + `--vram-reserve-mib 400` (live config before the LRU change) | 11.5 / 17.0 / 14.3 and 11.1 / 17.0 / 13.7 | `raw/lru/dd_e0.log` |
| 2026-10-12 05:33 | LRU 3,800 slots, tier 0.2 GiB, cold start | 12.7 / 17.5 / 14.3 and 12.6 / 16.4 / 14.5 | `raw/lru/dd_e1.log` |
| 2026-10-12 05:44 | LRU 4,400 + preload by profile rank | 13.7 / 17.3 / 15.9 and 13.3 / 17.9 / 15.5 | `raw/lru/dd_e2.log` |
| 2026-10-12 05:44 | LRU 5,600 + preload | 16.1 / 19.0 / 16.2 and 15.9 / 18.8 / 15.9 | `raw/lru/dd_e3.log` |
| 2026-10-12 05:59 | + SSD miss reads split in 4 parallel preads (negative, removed) | 15.5 / 17.6 / 15.1 and 15.2 / 16.9 / 15.6 | `raw/lru/dd_e5.log` |
| 2026-10-12 06:08 | + `MAC_PULL_NOCACHE=1` (**final**) | **16.9 / 19.0 / 16.3** and 16.8 / 19.0 / 15.8 in this run; **16.8 / 19.0 / 16.3** and **16.6 / 19.0 / 16.0** in `final_lru.log` | `raw/lru/dd_e7.log`, `raw/lru/final_lru.log` |

Negative results measured on the way (config, log in `results/raw/mini-shim-logs/` or `raw/decode/`): pool QoS + in-call spin, early pull started
from the CE push, 256K read chunks, look-ahead warm into a pread queue (engine died, root-caused, no gain after the fix), bigger locked RAM
tier (locked RAM displaces file cache 1:1), `--kv-resident 20480`, `--spec 6`, router look-ahead, SSD miss reads split x4, co-occurrence predictor
(9-16 % coverage, gate 80 %: not built), whole-resident-layer placement (wash at 6 layers, worse at 10+).

## 4. Prefill progression (tok/s by prompt tokens)

Source paths are relative to `results/raw/`.

| date (UTC) | change | prefill | source |
|---|---|---|---|
| 2026-10-11 17:45 | every cold expert streamed over USB per 2,048-token chunk (138 s per chunk) | 9.5 @2,562 / 10.8 @4,473 / 12.0 @6,991 | `mini-shim-logs/pf_F1.log`, `pf_F1_8k.log` |
| 2026-10-11 21:00 | same, after the decode work | 14.4 @3,467 | `mini-shim-logs/pf_WF6_m.log` |
| 2026-10-11 22:10 | Mac Metal cold-expert prefill lane (`STRATA_PF_MAC`) | 128.6 @3,467 | ledger row 2026-10-11T22:10Z (engine log lines not kept) |
| 2026-10-11 23:55 | fp16 partial sums, F_NOCACHE preads, per-prompt slot sets | 50 / 109 / 139 / 156 @569 / 2,753 / 4,473 / 7,348 | ledger row 2026-10-11T23:55Z |
| 2026-10-12 00:10 | pipelined USB transfers (h2d 228 -> 353 MB/s, d2h 154 -> 328 MB/s) | 47 / 121 / 140 / 154-160 @569 / 2,909 / 4,473 / 7,348 (SSD-bound) | `mini-shim-logs/final_pp.out`, ledger row 2026-10-12T00:10Z |
| 2026-10-12 01:10 | layer-major chunk-group lockstep G=3 | 231 @3,783, 178 @8,218, 363 @17,610 | `mini-shim-logs/ls_final.log` |
| 2026-10-12 03:22 | even chunks, cross-group read-ahead, fp16-I/O Metal prefill | 39 @569, 260-279 @3,783, 364 @6,991, 401 @7,905, 311 @8,218, 383 @17,610 | `mini-shim-logs/pf500_final.log` |
| 2026-10-12 04:26 | G=4 + `--vram-reserve-mib 400` | 259-287 @3,783, 373-414 @7,905, 316-318 @8,218, 385 @17,610 | `raw/g4/g4v_fix1.log`, `raw/g4/dc_g4c.log` |
| 2026-10-12 06:12 | final (LRU hits read in place by the prefill lane, prompt sets borrowed from the LRU) | 277-330 @3,783, 383-422 @7,905, 346 @8,218, 392 @17,610, 378 @33,253 | `raw/lru/final_lru.log` |

Limits found with measurements (`results/raw/duplex/`, `results/raw/pf500/`, plan entries 2026-10-12T00:10Z and 03:40Z): the USB link is half
duplex through the bridge (simultaneous arm drops the upload; 104.0 vs 104.2 MB/s sequential); prompts <= 4K tokens are SSD-bound
(477 MB of cold blobs per layer at ~2.0 of 2.41 GB/s); at G=3 one chunk-layer costs 96 ms (link 58, front 29, host ~8) against the 85 ms that 500 tok/s needs.

## 5. Memory and placement

| item | value | source |
|---|---|---|
| experts | 24,576 (48 layers x 512), 1,862,400 B each = 45,770,342,400 B, `exl3-k3-mul1-v1` | `results/pack/exl3_pack.json` |
| expert slots in VRAM | 7,400 (12.84 GiB), pre-filled from the profile in 64.1 s | `raw/lru/final_lru.log` |
| VRAM free after load | 716 MiB (`--vram-reserve-mib 400`) | same |
| Metal LRU | 5,600 slots x 1,867,776 B = 10.46 GB, mlocked, preloaded by profile rank in 4.8 s | same |
| static RAM tier | 0.2 GiB (115 blobs), was 5.1 GiB | plan entry 06:25Z |
| Metal working-set cap on the mini | 11.84 GiB (`recommendedMaxWorkingSetSize`); this, not RAM, bounds the LRU | `raw/lru/ws.m`, plan entry 06:25Z |
| engine RSS / kernel wired at READY | 10.57 GB RSS (`ps`), 11.6 GB wired | `raw/lru/final_lru.log` |
| 200K KV, k8v4 | 816 B/cell/layer, 2.61 GB; costs 1,308 of the 9,108 slots that fit without it | `results/campaign/GOAL.md` (Vision + 200K plan) |

## 6. Kernel and library receipts (card-free unless marked)

| component | result | source |
|---|---|---|
| EXL3 decode (reconstruct) on the 3090 | bit-exact, 50/50 | `results/kernels/m1_receipt.json` |
| exllamav3 GEMV via tgcu | 27/27, rel RMS <= 5.6e-4 (m 1/3/8); GEMM 69/69, rel RMS <= 1.1e-3 | `gemv_receipt.json`, `gemm_receipt.json` |
| fused MoE decode (`exl3_moe_coop`) | 11/11, rel RMS <= 8.1e-4; 108.8 us per layer at bsz 1 | `moe_receipt.json` |
| expert pack -> VRAM MoE | 12/12 (rel RMS <= 6.2e-4); device plan kernel 14/14 | `gpu_receipt.json`, `plan_receipt.json` |
| Strata EXL3 GPU arms (decode, prefill) | see receipts | `exl3s_receipt.json`, `exl3s_pf_receipt.json` |
| NEON CPU tier (`libexl3cpu`) | 106/106: decode + had_in bit-exact (K 1-8 x 3 codebooks), GEMV rel RMS <= 5.6e-7 | `cpu_receipt.json`, `cpu_run.log` |
| Metal lib (`libexl3metal`) vs NEON | all pass: rel RMS 4.5e-5 to 7.3e-5 (gate 1e-3), negatives detected, deterministic, thread-safe | `metal_test_run.log`, `metal_test_receipt.json` |
| Metal throughput | decode wall m=1 E=10/13/20: 0.659 / 0.793 / 1.107 ms; prefill 129 ms/layer @2,048 (1.56 TFLOPS), 344 ms @8,192 (2.34 TFLOPS) | `metal_bench_run.log` |

Those receipts were produced on 2026-10-10 / 11 by the scripts in `exl3/` and `exl3/strata_m4/` on the original machine. The GPU ones need the card
and were not re-run for this repo; the NEON and Metal ones were re-run on 2026-10-12, see section 8.

## 7. The weights

`results/pack/exl3_pack.json` holds, per layer, the sha256 of that layer's 953,548,800-byte section of `experts.bin`; all 48 layers are marked
`turboderp/Qwen3.8-Flash-Next-exl3@69e33439 3.05bpw_h5_ng5`. `results/pack/td_verify.log` is the full `pack_td.py verify` run of 2026-10-11 (every
layer: 6 experts re-fetched from Hugging Face and byte-compared, section sha256 equal to the manifest; final line `VERIFY OK`).

## 8. Reproducibility checks run on 2026-10-12 (card-free)

Run by `scripts/verify-card-free.sh` on the mini; raw output in [`results/verification/`](../results/verification/).

| check | result |
|---|---|
| Strata patches 0001-0004 `git apply --check` + apply, in order, on a fresh `git archive v0.1.41` (`fb58e0db`) | all four apply; resulting files byte-identical to the deployed trees (24/24 files) |
| the same series on a fresh public clone of `Niko1221/Strata` tag `v0.1.41` (driver Mac) | tag resolves to `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`; vision patch applies |
| `libexl3metal.dylib` rebuilt from this repo (driver Mac, `exl3/metal/build.sh`) | sha256 `de43149055d8...` = the deployed dylib, byte for byte |
| `libexl3metal` test on the mini with the rebuilt dylib + rebuilt `libexl3cpu` | ALL PASS (same checks as the original receipt) |
| `libexl3cpu.dylib` rebuilt (`exl3/cpu/build.sh`, SDK 26.5 or 26.0) | sha256 `9fe9cab5491e...` = the deployed dylib, byte for byte (an absolute `-o` path changes the bytes: clang puts it in the install name; the script builds with a relative name; SDK 15.x differs in 52 bytes) |
| `test_exl3_cpu.py` with the rebuilt dylib | ALL PASS 106/106 (the real-expert cases read 6 experts of layer 24 from the pack instead of the retired encoder fixture; `results/verification/cpu_receipt_rebuilt.json`) |
| `configs/render.sh` on the three config templates with the mini's `$HOME` | byte-identical to the live `~/shim/strata-exl3-full512-{final-g4-lru,final-g4,final}.json` |
| `pack_td.py index` | regenerated headers byte-identical to the existing index (sha256 `2dc6b0ce...`, `0ea6ec8c...`) |
| `pack_td.py sync 24` into a temp pack (0.95 GB fetched from the pinned revision, 76 s) | section sha256 `80eb9002da51...` = manifest layer 24; NEON forward vs reference rel RMS 5.2e-4 |
| `exl3_strata_pf.fatbin`, `exl3_strata_recon.fatbin` rebuilt with nvcc 13.4 in the container | sha256 identical to the deployed fatbins |
| `exl3_strata.fatbin` rebuilt | its id depends on the build path: built from `/Users/sero/exl3` it is **byte-identical** to the deployed one (`d19f0e17...`), from `/build/exl3` it is `82e480e5...` (same size 662,480 B; the earlier three different hashes were three different directories). The Docker build uses `/Users/sero/exl3` ([reproduce.md](reproduce.md#10-one-click-what-was-run)) |
| `libstrata_exl3.dylib` host library rebuilt (`exl3/strata_m4/build.sh`) | same size (4,450,936 B) and exported symbols as the deployed one; sha differs because it embeds the fatbin above |
| `tgcu_builtin.fatbin` / `tgcu_libc.cubin` rebuilt | **byte-identical** to `shim/prebuilt/` and the deployed files ([reproduce.md](reproduce.md#10-one-click-what-was-run)): `nvcc -std=c++17 -O3 -Xptxas -v --fatbin -arch=sm_86 -no-compress -x cu tgcu_builtin.cu` (an earlier try without `-Xptxas -v` differed only in the command line stored in the fatbin header, not in the path) and `ptxas -arch=sm_86 -c tgcu_libc.ptx` |
| `libcudart.dylib` shim compiled from `shim/tgcudart.cpp` + `tgcublas.cpp` | compiles; exported symbols identical to the deployed one (136); 121,488 B vs 138,944 B (build flags not recorded; about 15 flag sets tried, none matched the size). The source is the `hs2` snapshot of `tgcudart.cpp`: the copy left in the mini's `~/shim` predates the pre-armed handshake and lacks six exported `tgcuHs*` symbols the deployed dylib has |
| `strata-vision-metal` built from the public tag + `patches/vision` + llama.cpp `3cf03257f` (38 s) | builds; same size as the deployed binary (8,077,952 B), sha256 differs (signature); **not run** against the mmproj |
| full engine rebuild (CMake configure in the container, `macbuild.py`, ninja) | device half re-run in the Docker container, host half on a driver Mac (not the mini); never run: see [reproduce.md section 10](reproduce.md#10-one-click-what-was-run) |
| Docker build bundle (`docker compose build --no-cache build`, 3 min 4 s, then `docker compose run --rm build`) | EXL3 fatbins + `tgcu_builtin.fatbin` + `tgcu_libc.cubin` byte-identical to the deployed files; Strata engine device code: 63 fatbins, 59 byte-identical to the deployed ones in every one of four builds, 4 vary run to run (nvcc anonymous-namespace ids) |
| `scripts/test-gates.sh` | ALL GATES OK: wrong hash, corrupted byte, unknown command and a failing `verify-card-free.sh` step all exit nonzero (the old `pack_td.py verify` printed `VERIFY FAILED` and exited 0) |
| `./run.sh --check` on a driver Mac (empty install root) | 53 ok, 13 warnings, 0 failures; engine host build with LLVM 22 only (Apple clang 17 fails) |

## 9. Exact, rounding-level, approximate

- **Exact:** the weights (every routed expert of every layer: the checkpoint's trellis and scale tensors byte for byte, the three `mul1` scalars one common value (checked for all 73,728
  tensors); nothing pruned or masked); the GPU
  EXL3 kernels (bit-exact decode, GEMV/MoE within the gates in section 6); the NEON path; text parity between decode transport variants
  (`HS_PRE`, `HS_DEFER`, `MAC_PULL`, early pull: byte-identical outputs on 6/6 requests, ledger rows 2026-10-11T18:55Z to 20:50Z).
- **Rounding-level:** with `STRATA_EXL3_METAL_SPLIT=1` (in the final config) a layer's experts are summed as `y = A + B` in fp32, where the
  in-place/miss partition depends on the LRU contents, so greedy text depends on LRU state. Final-config texts for the 3 probes diverge from
  the previous config's at character 451 / 409 / 49 (the previous config diverges from itself run to run at 451 / 217 / 49); all coherent, same
  acceptance. The variant without `SPLIT` is bit-exact and ~8-11 % slower. The Mac prefill lane returns fp16-rounded partial sums
  (`STRATA_PF_MAC_Y16=1`); an earlier A/B had the fp16 lane diverge from the fp32 lane at character 127 of a 48-token comparison.
- **Not measured:** KL or top-1 agreement against stock exllamav3 for any of these configs; GPQA Diamond; tool-calling benchmarks; concurrency above 1;
  the final config on the 190K needle; a vision run after the LRU change as a raw log.
