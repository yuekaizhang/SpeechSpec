# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark speculative decoding for speech LLMs (audio -> text ASR) in sglang.

End-to-end: launches an sglang server (with or without a draft model), decodes
LibriSpeech test-clean or AISHELL test, and reports WER/CER, latency, RTF,
throughput, and speculative accept length (completion_tokens / spec_verify_ct).

The speculative algorithm is inferred from the draft checkpoint's config.json:
`DFlashDraftModel` -> DFLASH (draft tokens = training block_size);
anything EAGLE3-shaped -> EAGLE3 (default steps 3 / topk 1 / draft tokens 4).

Example usage:
    # Qwen3-Omni baseline on LibriSpeech test-clean
    python3 benchmark_speech_llm.py \
        --target-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --dataset librispeech --output-dir results/omni_baseline

    # Qwen3-Omni + EAGLE3 (multi-domain audio) draft
    python3 benchmark_speech_llm.py \
        --target-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --draft-model yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio \
        --dataset librispeech --output-dir results/omni_eagle3

    # Qwen3-Omni + DFlash (text+audio mix, block 8) draft
    python3 benchmark_speech_llm.py \
        --target-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --draft-model yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8 \
        --dataset librispeech --output-dir results/omni_dflash

    # Qwen2-Audio AISHELL SFT baseline / EAGLE3 / DFlash block-4 on AISHELL test
    python3 benchmark_speech_llm.py \
        --target-model yuekai/qwen2_audio_aishell_sft \
        --dataset aishell --output-dir results/q2a_baseline
    python3 benchmark_speech_llm.py \
        --target-model yuekai/qwen2_audio_aishell_sft \
        --draft-model yuekai/qwen2_audio_aishell_sft_eagle3 \
        --dataset aishell --output-dir results/q2a_eagle3
    python3 benchmark_speech_llm.py \
        --target-model yuekai/qwen2_audio_aishell_sft \
        --draft-model yuekai/qwen2_audio_aishell_sft_dflash_block4 \
        --dataset aishell --output-dir results/q2a_dflash

    # Against an already-running server (skip launching)
    python3 benchmark_speech_llm.py --server http://127.0.0.1:30000 \
        --target-model qwen3-omni --dataset librispeech --output-dir out
"""
import argparse
import asyncio
import base64
import io
import json
import os
import re
import signal
import string
import subprocess
import sys
import time

import aiohttp
import numpy as np
import soundfile as sf

DATASET_PRESETS = {
    "librispeech": dict(
        hf_dataset="openslr/librispeech_asr",
        subset="clean",
        split="test",
        text_column="text",
        prompt="Transcribe the English audio into text.",
        metric="wer",  # uppercase + strip punctuation on both sides
    ),
    "aishell": dict(
        hf_dataset="carlot/AIShell",
        subset=None,
        split="test",
        text_column="transcription",
        prompt="Detect the language and recognize the speech: <|zh|>",
        metric="cer",
    ),
}


# ----------------------------- text metrics -----------------------------
def _edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[m]


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_english(s):
    return re.sub(r"\s+", " ", s.upper().translate(_PUNCT_TABLE)).strip()


def error_rate(ref, hyp, metric):
    if metric == "wer":
        r, h = normalize_english(ref).split(), normalize_english(hyp).split()
    else:  # cer, character-level, spaces stripped
        r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    if not r:
        return 0.0 if not h else float("inf"), 0
    return _edit_distance(r, h) / len(r), len(r)


# ----------------------------- audio encode -----------------------------
def encode_audio_b64(audio_bytes):
    """Raw container bytes -> 16 kHz mono WAV base64 + duration (s)."""
    arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        n = int(round(len(arr) / sr * 16000))
        arr = np.interp(
            np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr
        ).astype(np.float32)
        sr = 16000
    dur = len(arr) / sr
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii"), dur


# ----------------------------- draft inspection -----------------------------
def load_draft_config(draft):
    p = os.path.join(draft, "config.json")
    if os.path.isfile(p):
        return json.load(open(p))
    from huggingface_hub import hf_hub_download

    return json.load(open(hf_hub_download(draft, "config.json")))


def spec_args_for_draft(draft, args):
    """Infer --speculative-* server args from the draft checkpoint config."""
    cfg = load_draft_config(draft)
    archs = cfg.get("architectures") or []
    if any("DFlash" in a for a in archs):
        block = args.num_draft_tokens or cfg.get("block_size", 8)
        return "DFLASH", [
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", draft,
            "--speculative-num-draft-tokens", str(block),
        ]
    # EAGLE3 family
    return "EAGLE3", [
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", draft,
        "--speculative-num-steps", str(args.num_steps),
        "--speculative-eagle-topk", str(args.eagle_topk),
        "--speculative-num-draft-tokens", str(args.num_draft_tokens or 4),
    ]


# ----------------------------- server -----------------------------
def launch_server(args, spec_cli):
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.target_model,
        "--served-model-name", "speech-llm",
        "--tp", str(args.tp),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "bfloat16",
        "--attention-backend", "fa3",
        "--mem-fraction-static", str(args.mem_fraction),
        "--grammar-backend", "none",
        "--cuda-graph-max-bs-decode", str(max(args.concurrency, 1)),
    ] + spec_cli
    if spec_cli and args.context_length:
        cmd += ["--context-length", str(args.context_length)]
    env = dict(os.environ)
    # Draft configs may declare a shorter max_position_embeddings than the
    # target; allow serving at the target's context length anyway.
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    log = open(args.server_log, "w")
    print(f"Launching sglang server (log: {args.server_log}) ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    deadline = time.monotonic() + args.server_timeout
    import urllib.request

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server died; see {args.server_log}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=3)
            print("Server is up.", flush=True)
            return proc
        except Exception:
            time.sleep(5)
    raise RuntimeError("server did not come up in time")


def kill_server(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()


# ----------------------------- client -----------------------------
async def decode_one(session, url, prompt, audio_b64, max_tokens, sem):
    async with sem:
        payload = {
            "model": "speech-llm",
            "temperature": 0,
            "max_tokens": max_tokens,
            "return_meta_info": True,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio_url",
                     "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        t0 = time.monotonic()
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                latency = time.monotonic() - t0
                if "choices" in data:
                    usage = data.get("usage") or {}
                    meta = data["choices"][0].get("meta_info") or {}
                    return data["choices"][0]["message"]["content"], latency, {
                        "completion_tokens": usage.get("completion_tokens"),
                        "spec_verify_ct": meta.get("spec_verify_ct"),
                    }
                return f"ERROR: {json.dumps(data)[:200]}", latency, {}
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}", time.monotonic() - t0, {}


async def decode_all(server, prompt, clips, max_tokens, concurrency):
    url = f"{server}/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = [decode_one(session, url, prompt, c, max_tokens, sem) for c in clips]
        return await asyncio.gather(*tasks)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--draft-model", default=None,
                    help="HF repo or local checkpoint dir; omit for baseline")
    ap.add_argument("--dataset", choices=sorted(DATASET_PRESETS), required=True)
    ap.add_argument("--num-samples", type=int, default=0, help="0 = full test set")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    # spec-decoding knobs (EAGLE3 defaults follow the report: steps 3/topk 1/draft 4)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--eagle-topk", type=int, default=1)
    ap.add_argument("--num-draft-tokens", type=int, default=None,
                    help="default: draft block_size (DFLASH) or 4 (EAGLE3)")
    # server knobs
    ap.add_argument("--server", default=None,
                    help="use an existing server instead of launching one")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=8192,
                    help="context cap when serving with a draft (0 = target default)")
    ap.add_argument("--server-timeout", type=int, default=1800)
    ap.add_argument("--server-log", default=None)
    args = ap.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    os.makedirs(args.output_dir, exist_ok=True)
    if args.server_log is None:
        args.server_log = os.path.join(args.output_dir, "server.log")

    # ---- dataset ----
    from datasets import load_dataset
    from datasets.features import Audio

    print(f"Loading {preset['hf_dataset']} split={preset['split']} ...", flush=True)
    if preset["subset"]:
        ds = load_dataset(preset["hf_dataset"], preset["subset"], split=preset["split"])
    else:
        ds = load_dataset(preset["hf_dataset"], split=preset["split"])
    ds = ds.cast_column("audio", Audio(decode=False))
    n = len(ds) if args.num_samples <= 0 else min(args.num_samples, len(ds))
    print(f"Encoding {n} clips ...", flush=True)
    clips, refs, durs = [], [], []
    for i in range(n):
        row = ds[i]
        b64, dur = encode_audio_b64(row["audio"]["bytes"])
        clips.append(b64)
        durs.append(dur)
        refs.append(row[preset["text_column"]])

    # ---- server ----
    proc = None
    server = args.server
    algo = "none"
    spec_cli = []
    if args.draft_model:
        algo, spec_cli = spec_args_for_draft(args.draft_model, args)
        print(f"Draft: {args.draft_model} -> {algo} {spec_cli[4:]}", flush=True)
    if server is None:
        if not args.context_length or not args.draft_model:
            args.context_length = 0
        proc = launch_server(args, spec_cli)
        server = f"http://127.0.0.1:{args.port}"

    try:
        if args.warmup > 0:
            asyncio.run(decode_all(server, preset["prompt"], clips[: args.warmup],
                                   args.max_tokens, args.concurrency))
        t0 = time.monotonic()
        results = asyncio.run(decode_all(server, preset["prompt"], clips,
                                         args.max_tokens, args.concurrency))
        wall = time.monotonic() - t0
    finally:
        if proc is not None:
            kill_server(proc)

    # ---- metrics ----
    rows, errors = [], 0
    tot_edits = tot_ref = 0
    tot_ct = tot_vc = 0
    tot_lat = 0.0
    accepts = []
    for i, (hyp, latency, stats) in enumerate(results):
        if hyp.startswith("ERROR:"):
            errors += 1
            er, rl = float("inf"), 0
        else:
            er, rl = error_rate(refs[i], hyp, preset["metric"])
            tot_edits += er * rl
            tot_ref += rl
        ct = stats.get("completion_tokens") or 0
        vc = stats.get("spec_verify_ct")
        acc = (ct / vc) if (vc and ct) else None
        if acc is not None:
            accepts.append(acc)
            tot_vc += vc
        tot_ct += ct
        tot_lat += latency
        rows.append({
            "idx": i, "ref": refs[i], "hyp": hyp,
            "audio_dur_s": round(durs[i], 2), "latency_s": round(latency, 4),
            preset["metric"]: round(er, 4) if er != float("inf") else None,
            "completion_tokens": ct or None, "spec_verify_ct": vc,
            "accept_len": round(acc, 4) if acc else None,
        })

    with open(os.path.join(args.output_dir, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_audio = sum(durs)
    err_pct = 100.0 * tot_edits / max(tot_ref, 1)
    lines = [
        "=== Speech LLM ASR Benchmark ===",
        f"Target:       {args.target_model}",
        f"Draft:        {args.draft_model or 'None (baseline)'}",
        f"Spec algo:    {algo}",
        f"Dataset:      {args.dataset} ({n} utts)   Concurrency: {args.concurrency}",
        "",
        f"{preset['metric'].upper()}:          {err_pct:.2f}%",
        f"Errors:       {errors}",
        f"Wall time:    {wall:.1f}s",
        f"Total audio:  {total_audio:.1f}s (overall RTF {wall / max(total_audio, 1e-9):.4f})",
        f"Mean latency: {tot_lat / max(n, 1):.3f}s",
        f"Throughput:   {n / wall:.1f} utt/s",
    ]
    if accepts:
        lines += [
            f"Accept (micro): {tot_ct / tot_vc:.3f}",
            f"Accept (macro): {sum(accepts) / len(accepts):.3f}",
        ]
    summary = "\n".join(lines)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
