#!/usr/bin/env python3
"""完整验证: Taobao prompt × bs=1/2/3/4 sync barrier + max_tokens=4096."""

import json
import re
import subprocess
import sys
import threading
import time

ENDPOINT = "http://localhost:8066/v1/chat/completions"
MODEL = "Qwen3.5-9B"
MAX_TOKENS = 4096

MESSAGES = json.load(open("/root/rtp-llm/taobao_messages.json"))


def call(out, idx, payload):
    p = subprocess.run(
        [
            "curl",
            "-s",
            ENDPOINT,
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=900,
    )
    try:
        out[idx] = json.loads(p.stdout)["choices"][0]["message"]["content"]
    except Exception as e:
        out[idx] = f"[fail: {e}: {p.stdout[:200]}]"


def send_n(n, max_tokens=MAX_TOKENS):
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": MESSAGES,
            "max_tokens": max_tokens,
            "temperature": 0,
            "n": 1,
        },
        ensure_ascii=False,
    )
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = []

    def worker(i):
        barrier.wait()
        call(out, i, payload)

    for i in range(n):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return out


REPEAT = re.compile(r"(.{8,80}?)\1{4,}")


def detect_anomaly(text):
    if not text:
        return ["NULL"]
    issues = []
    if REPEAT.search(text):
        issues.append("REPEAT")
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        if tok in text:
            issues.append(f"LEAK({tok})")
    return issues


def main():
    print("=" * 60)
    print("bs=1 baseline")
    print("=" * 60)
    t0 = time.time()
    [ref] = send_n(1)
    print(
        f"  done {time.time()-t0:.1f}s, len={len(ref)}, anomaly={detect_anomaly(ref) or 'none'}"
    )

    total_pass = 0
    total_fail = 0
    for conc in [2, 3, 4]:
        time.sleep(2)
        print(f"\n--- bs={conc} sync barrier ---")
        t0 = time.time()
        outs = send_n(conc)
        print(f"  done {time.time()-t0:.1f}s")
        for i, o in enumerate(outs):
            if o == ref:
                total_pass += 1
                print(f"  slot {i}: MATCH (len={len(o)})")
            else:
                total_fail += 1
                if not o or "[fail" in str(o):
                    print(f"  slot {i}: ERROR {o[:200] if o else 'None'}")
                else:
                    div = next(
                        (j for j in range(min(len(ref), len(o))) if ref[j] != o[j]),
                        min(len(ref), len(o)),
                    )
                    print(
                        f"  slot {i}: MISMATCH at char {div} (ref={len(ref)} got={len(o)})"
                    )

    print(f"\n{'='*60}")
    print(f"FINAL: {total_pass}/{total_pass+total_fail} MATCH, {total_fail} FAIL")
    print(f"{'='*60}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
