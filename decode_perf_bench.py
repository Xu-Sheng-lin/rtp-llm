#!/usr/bin/env python3
"""Decode 性能 benchmark: 控制 prefill 影响,测 decode 吞吐.

策略:
- 短 prompt (~200 字符,~100 token) 最小化 prefill 时间
- 固定 max_tokens=512,强制长 decode
- 用 stream 模式获取 first_token_time,精确分离 prefill / decode
- bs=1/4/8 三档,看不同并发下 decode throughput
"""

import json
import subprocess
import sys
import time
from statistics import mean, stdev

ENDPOINT = "http://localhost:8066/v1/chat/completions"
MODEL = "Qwen3.5-9B"
MAX_TOKENS = 512

# 短 prompt: 触发足够长的 decode,但 prefill 开销小
PROMPT = "请用中文写一篇关于人工智能的短文,介绍 AI 的发展历史、当前应用、未来挑战三个方面。每个方面 2-3 段。注意结构清晰。"


def send_stream(prompt, max_tokens=MAX_TOKENS):
    """Stream 模式: 返回 (first_token_time, total_time, output_text)."""
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        ensure_ascii=False,
    )

    t0 = time.time()
    proc = subprocess.Popen(
        [
            "curl",
            "-s",
            "-N",
            ENDPOINT,
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc.stdin.write(payload)
    proc.stdin.close()

    first_token_time = None
    output = []
    n_tokens = 0
    for line in proc.stdout:
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if first_token_time is None:
                    first_token_time = time.time() - t0
                output.append(content)
                n_tokens += 1
        except Exception:
            continue
    proc.wait()
    total_time = time.time() - t0
    return first_token_time, total_time, "".join(output), n_tokens


def send_concurrent_stream(prompt, n, max_tokens=MAX_TOKENS):
    """并发 N 个 stream 请求."""
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        ensure_ascii=False,
    )

    t0 = time.time()
    procs = []
    for _ in range(n):
        p = subprocess.Popen(
            [
                "curl",
                "-s",
                "-N",
                ENDPOINT,
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        p.stdin.write(payload)
        p.stdin.close()
        procs.append((p, [], None, 0, t0))

    results = []
    for p, output, ftt, nt, start_t in procs:
        first_token_time = None
        n_tokens = 0
        for line in p.stdout:
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if first_token_time is None:
                        first_token_time = time.time() - start_t
                    output.append(content)
                    n_tokens += 1
            except Exception:
                continue
        p.wait()
        total_time = time.time() - start_t
        results.append((first_token_time, total_time, "".join(output), n_tokens))
    return results


def warmup(n=2):
    print(f"[WARMUP] {n} requests...")
    for i in range(n):
        ftt, tt, _, nt = send_stream(PROMPT, max_tokens=64)
        print(f"  warm {i+1}: ftt={ftt:.2f}s total={tt:.2f}s tokens={nt}")


def bench_bs1(repeats=3):
    print(f"\n--- bs=1, repeats={repeats} ---")
    runs = []
    for i in range(repeats):
        ftt, tt, out, nt = send_stream(PROMPT)
        decode_time = tt - ftt
        decode_tps = (nt - 1) / decode_time if decode_time > 0 else 0
        runs.append((ftt, tt, nt, decode_tps))
        print(
            f"  run {i+1}: ftt={ftt:.3f}s total={tt:.3f}s tokens={nt} decode_tps={decode_tps:.1f}"
        )
    return runs


def bench_concurrent(conc, repeats=2):
    print(f"\n--- bs={conc}, repeats={repeats} ---")
    all_runs = []
    for i in range(repeats):
        results = send_concurrent_stream(PROMPT, conc)
        run_data = []
        for slot, (ftt, tt, _, nt) in enumerate(results):
            decode_time = tt - ftt
            decode_tps = (nt - 1) / decode_time if decode_time > 0 else 0
            run_data.append((ftt, tt, nt, decode_tps))
            print(
                f"  run {i+1} slot {slot}: ftt={ftt:.3f}s total={tt:.3f}s tokens={nt} decode_tps={decode_tps:.1f}"
            )
        all_runs.append(run_data)
    return all_runs


def summarize(label, runs_list):
    """runs_list: list of (ftt, total, tokens, decode_tps) tuples."""
    if not runs_list:
        return
    ftts = [r[0] for r in runs_list if r[0]]
    decodes = [r[3] for r in runs_list if r[3] > 0]
    tokens = [r[2] for r in runs_list]
    print(
        f"  [{label}] ftt mean={mean(ftts):.3f}s | decode_tps mean={mean(decodes):.1f}"
        f" stdev={stdev(decodes) if len(decodes)>1 else 0:.1f}"
        f" | tokens mean={mean(tokens):.0f}"
    )


def main():
    print("=" * 70)
    print("Decode performance benchmark")
    print("=" * 70)
    print(f"prompt chars: {len(PROMPT)}")
    print(f"max_tokens: {MAX_TOKENS}")

    warmup(2)

    bs1_runs = bench_bs1(repeats=3)
    print("\n[Summary bs=1]")
    summarize("bs=1", bs1_runs)

    bs4_runs = bench_concurrent(4, repeats=2)
    flat_bs4 = [r for run in bs4_runs for r in run]
    print("\n[Summary bs=4 per-slot]")
    summarize("bs=4", flat_bs4)
    # Aggregate throughput
    total_tokens_bs4 = [sum(r[2] for r in run) for run in bs4_runs]
    elapsed_bs4 = [max(r[1] for r in run) for run in bs4_runs]
    agg_tps_bs4 = [t / e for t, e in zip(total_tokens_bs4, elapsed_bs4)]
    print(
        f"  [bs=4 aggregate throughput] mean={mean(agg_tps_bs4):.1f} tps over all 4 slots"
    )

    bs8_runs = bench_concurrent(8, repeats=1)
    flat_bs8 = [r for run in bs8_runs for r in run]
    print("\n[Summary bs=8 per-slot]")
    summarize("bs=8", flat_bs8)
    total_tokens_bs8 = [sum(r[2] for r in run) for run in bs8_runs]
    elapsed_bs8 = [max(r[1] for r in run) for run in bs8_runs]
    agg_tps_bs8 = [t / e for t, e in zip(total_tokens_bs8, elapsed_bs8)]
    print(
        f"  [bs=8 aggregate throughput] mean={mean(agg_tps_bs8):.1f} tps over all 8 slots"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
