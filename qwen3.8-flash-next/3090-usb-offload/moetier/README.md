# moetier

Records for the [moetier](https://github.com/sybil-solutions) placement/scheduling simulator used to project this system before building each lever
(the simulator itself is not in this repo; the scripts take its checkout with `--moetier`).

- `registry/` - hardware (`chestnut-3090-usb-macmini-m4`), engine (`strata-tgcu-exl3`: per-layer USB handshake 3.5 ms, 0.25 ms per distinct cold expert), model
  (`qwen3.8-flash-next-exl3-3.05bpw`) and recipe records. The recipe is **calibrated to the first full-512 end-to-end run of 2026-10-11** (17 whole-resident layers, 8,770 slots): the simulation gives 5.68 tok/s
  against 5.1-5.6 measured. It was not re-calibrated to the final configuration.
- `levers_cal.py` / `levers_cal_out.md` - the levers table produced from that calibration (Metal lane, hidden handshake, +4,000 slots, ...); the routing input is the P256 calibration trace
  (`trace_p256.npy`, 20 MB, not committed) plus `routed_counts.npy`, with a q = 0.2 tail for non-kept experts because no full-512 routing trace existed.
- `levers_strata.py` - the earlier, uncalibrated levers run.

The model predicted prefill 290-780 tok/s on 2026-10-11 and was wrong by 25-80x (it assumed cold experts reach the GPU at NVMe speed in chunks; on this rig they had to cross a 190 MB/s link):
see `results/campaign/EXL3_PLAN-log.md`, 2026-10-11T17:45Z. Decode predictions held within +/-30 %.
