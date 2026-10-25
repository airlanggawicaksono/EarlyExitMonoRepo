"""Limited-preemptive (LP) feasibility probe on the board: pause a compiled LLaMA request at a token boundary.

Owner 2026-10-26: "yaudah testing lp cepat jangan banyak2 token gitu cara nya giamna terserah pastikan bsia di lp
dengan compiled model thx", "di jetson ya ini".

One process loads the Run 3 adapters (LLaMA and BERT, torch.compile per layer) and warms them up. Then:
  ref : one LLaMA request (exit 15, 32 tokens) runs without a pause.
  lp  : the same request pauses after token k, BERT requests run on the GPU, then LLaMA resumes to token 32.
It reports, per pause point: identical tokens (True/False), new dynamo graphs after warm-up (0 = no recompile),
the first token after resume vs the steady decode time, the KV-cache memory held while paused, and the pause cost.
Board only (never on the laptop). usage: python -m exp2.board.scripts.lp_probe OUT.json
"""
import json
import sys
import time

from exp2.serve import families

LLAMA_EXIT, BERT_EXIT, BERT_PER_PAUSE = 15, 23, 5
PAUSE_AT = (1, 8, 16, 31)     # tokens done when the pause starts (1 = right after the prefill)
MIN_AVAIL_MB = 400            # the board hung at ~85 MB MemAvailable; stop well before that


def avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return -1


def guard(tag):
    m = avail_mb()
    print(f"[lp_probe] {tag}: MemAvailable {m} MB", file=sys.stderr, flush=True)
    if m < MIN_AVAIL_MB:
        raise SystemExit(f"MemAvailable {m} MB < {MIN_AVAIL_MB} MB at {tag}")


def graphs():
    from torch._dynamo.utils import counters
    return {"unique_graphs": counters["stats"]["unique_graphs"], "frames_ok": counters["frames"]["ok"]}


def llama_tokens(llama, idx, pause_at=None, on_pause=None):
    """Run one request token by token. Returns (token ids, per-token ms, pause info)."""
    import torch

    st = llama.begin(idx, LLAMA_EXIT)
    ms, info = [], None
    for k in range(1, 33):
        t0 = time.perf_counter()
        llama.advance(st, k)
        llama.sync()
        ms.append((time.perf_counter() - t0) * 1e3)
        if k == pause_at:
            t1 = time.perf_counter()
            alloc = torch.cuda.memory_allocated() / 2**20
            on_pause()
            info = {"paused_ms": (time.perf_counter() - t1) * 1e3, "alloc_mb_at_pause": alloc,
                    "alloc_mb_after_others": torch.cuda.memory_allocated() / 2**20}
    return llama.finish(st, 32), ms, info


def bert_request(bert, idx):
    st = bert.begin(idx, BERT_EXIT)
    for c in bert.checkpoints(BERT_EXIT):
        bert.advance(st, c)
    out = bert.finish(st, BERT_EXIT)
    bert.sync()
    return out


def main(out_path):
    import torch

    t_start = time.perf_counter()
    compile_state = families.setup_compile(True)
    guard("start")
    llama, bert = families.get_adapter("llama"), families.get_adapter("bert")
    llama.load([LLAMA_EXIT], {"compile": True})
    llama.make_inputs(4, 0, "synthetic")
    guard("llama loaded")
    bert.load([BERT_EXIT], {"compile": True})
    bert.make_inputs(8, 0, "synthetic")
    guard("bert loaded")
    for _ in range(3):        # warm-up: every prompt/decode shape and the BERT layers get compiled here
        llama_tokens(llama, 0)
    for i in range(10):
        bert_request(bert, i % 8)
    guard("warm")
    g0 = graphs()
    ref, ref_ms, _ = llama_tokens(llama, 0)
    steady = sorted(ref_ms[1:])[len(ref_ms[1:]) // 2]   # median decode token (token 2..32)
    rows = []
    for p in PAUSE_AT:
        def on_pause():
            for i in range(BERT_PER_PAUSE):
                bert_request(bert, i)
        toks, ms, info = llama_tokens(llama, 0, pause_at=p, on_pause=on_pause)
        rows.append(dict(info, pause_at=p, same_tokens=toks == ref,
                         first_after_resume_ms=ms[p] if p < 32 else None,
                         rest_median_ms=sorted(ms[p:])[len(ms[p:]) // 2] if p < 32 else None))
        guard(f"pause_at {p}")
    idle_toks, idle_ms, idle = llama_tokens(llama, 0, pause_at=8, on_pause=lambda: time.sleep(1.0))
    g1 = graphs()
    res = {
        "compile": compile_state, "torch": torch.__version__, "llama_exit": LLAMA_EXIT, "bert_exit": BERT_EXIT,
        "bert_per_pause": BERT_PER_PAUSE, "prefill_ms": ref_ms[0], "decode_median_ms": steady,
        "decode_max_ms": max(ref_ms[1:]), "lp": rows,
        "idle_pause": dict(idle, pause_at=8, same_tokens=idle_toks == ref, first_after_resume_ms=idle_ms[8]),
        "graphs_after_warmup": g0, "graphs_at_end": g1,
        "new_graphs": g1["unique_graphs"] - g0["unique_graphs"],
        "gpu_cur_freq": open("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq").read().strip(),
        "wall_s": time.perf_counter() - t_start,
    }
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
