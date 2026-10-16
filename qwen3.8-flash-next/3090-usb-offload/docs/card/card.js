// Card text: measured numbers from the live config final-g4-lru (EXL3_PLAN.md 2026-10-12 log). Render: ./render.sh -> exl3-card.png
var CARD = {
  word: 'Sybil AI',
  credit: 'art · @samdape',
  tag: 'Measured',
  model: 'Qwen3.8-Flash-Next · EXL3 3bpw',
  sub: 'all 512 experts, exact · 1× RTX 3090 over USB + Mac mini M4 16 GB · Strata + Metal',
  stats: [['16–19', 'tok/s decode c1'], ['422', 'tok/s prefill @7.9K'], ['200K', 'context · needle ok'], ['vision', 'image in · Metal encoder']],
  detail: ['turboderp 3.05bpw', 'MTP spec 4', 'k8v4 KV', '7,400 experts in VRAM', '5,600 in mini RAM', 'rest on NVMe', '382 tok/s prefill @189K'],
  row: ['PCIe Gen3 x2 behind USB 3.2 10 Gb/s', 'updated 2026-10-12'],
  foot: '@0xSero',
};
