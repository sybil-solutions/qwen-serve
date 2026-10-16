# results/

Raw files behind the tables in [`docs/results.md`](../docs/results.md). Paths inside logs have `/Users/<user>` replaced by `$HOME`; nothing else was edited.

| directory | contents |
|---|---|
| `raw/lru/` | the final configuration: `final_lru.log` (restart, decode x2, prefill rows, 32K needle, decode after prefill), `dd_e0..e7.log` (one restart + decode x2 + `STRATA_DECODE_TIMING` lines per LRU lever: e0 = config before the change, e1 LRU 3,800, e2 4,400 + preload, e3/e4 5,600, e5 + split SSD reads, e6, e7 final), the scripts that produced them (`dd.sh`, `dd2.sh`, `final_lru.sh`), the simulators (`sim2.py`, `sim3.py`, `sim4.py`), `ws.m` (prints Metal's recommended working set), config variants `e*.json` |
| `raw/g4/` | G=4 bring-up: `final_g4.log` (32K + 190K needles), `lr_g4c_100k.log` (100K needle), `g4v_fix1.log` / `dc_g4c.log` (decode + prefill rows), `g4_s1_hang_excerpt.txt` (the hang dump), `pw_*.log` (whole-layer placement W=0/6/10/15), config variants |
| `raw/decode/` | decode levers of 2026-10-12 02:00-03:00Z: `dec_*.log` (baseline, split, early, look-ahead, spec 6, kv-resident, trace), `ssdb.py` (SSD microbench) and its output, `sim.py` (LRU simulation) |
| `raw/mini-shim-logs/` | the mini's `~/shim/*.log` for every run named in the ledger (`split_*` decode probes, `pf_*` and `pf500_*` prefill, `ls_*` lockstep, `dec_final.log`, `final_pp.out`, ...). The 33 MB `serve-engine.log` is not included |
| `raw/ref-full512/` | greedy reference texts of the first full-512 runs (`F1_*.txt` ... `F6m_lru1000_*.txt`) and their logs |
| `raw/duplex/`, `raw/pf500/` | USB duplex / pipelined-transfer microbench scripts, prefill timing helpers |
| `kernels/` | receipts of the kernel and library tests (GPU ones from 2026-10-10/11) |
| `pack/` | `exl3_pack.json` (per-layer sha256 manifest of `experts.bin`), `td_sync.log`, `td_verify.log` (`VERIFY OK`) |
| `provenance/` | the MTP extraction logs and manifest (tensor names, shards, byte ranges, sha256), the dense pack log |
| `verification/` | the card-free checks run for this repo on 2026-10-12 |
| `campaign/` | `ledger.jsonl` (one row per experiment), `EXL3_PLAN-log.md` (the campaign log, verbatim), `GOAL.md` (goal, status blocks, status table) |
