# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Benchmark speculative decoding for CosyVoice-style TTS LLMs in vLLM.

Measures wall time / throughput / speculative acceptance over a seed-tts style
dataset using zero-shot voice-clone prompting (the prompt-audio speech tokens
are given as an assistant prefix and the LLM continues them):

    <|sos|>{prompt_text + target_text}<|task_id|>{prompt speech ids}...

Only LLM token generation is benchmarked (no token2wav, no audio IO).

Example usage:
    # CosyVoice2 baseline (no speculative decoding)
    python3 benchmark_tts.py --target-model yuekai/cosyvoice2_llm \
        --dataset yuekai/seed_tts_cosy2 --dataset-split test_zh \
        --batch-size 1,8,16,64 --output-json cv2_baseline.json

    # CosyVoice2 + DSpark drafter
    python3 benchmark_tts.py --target-model yuekai/cosyvoice2_llm \
        --draft-model yuekai/cosyvoice2_llm_dspark \
        --dataset yuekai/seed_tts_cosy2 --dataset-split test_zh \
        --batch-size 1,8,16,64 --output-json cv2_dspark.json

    # CosyVoice3 + DSpark drafter (needs vllm-project/vllm#48932; the
    # repetition-penalty mirror is enabled automatically when
    # --repetition-penalty != 1.0)
    python3 benchmark_tts.py \
        --target-model yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF \
        --draft-model yuekai/cosyvoice3_llm_dspark \
        --dataset yuekai/seed_tts_zh_cosy3 --dataset-split test_zh \
        --top-k 15 --repetition-penalty 1.1 \
        --batch-size 1,8,16,64 --output-json cv3_dspark.json

The speculative method (dflash / dspark / eagle3) and the number of
speculative tokens are auto-derived from the draft checkpoint's config.json
(speculators format); override with --method / --num-spec-tokens if needed.
"""

import argparse
import json
import os
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PUNCTS = ['"', "(", ")", "“", "”", "‘", "（", "）", "'"]

# CosyVoice3 LLM prompts must start with this prefix (see the CosyVoice3
# chat usage); auto-applied when the target model looks like CosyVoice3.
COSYVOICE3_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def get_args():
    parser = argparse.ArgumentParser(
        description="Speculative-decoding benchmark for CosyVoice TTS LLMs"
    )
    parser.add_argument("--target-model", type=str, required=True,
                        help="target LLM (HF id or local path), e.g. "
                             "yuekai/cosyvoice2_llm")
    parser.add_argument("--draft-model", type=str, default=None,
                        help="speculators-format draft checkpoint (HF id or "
                             "local path). Omit for the no-speculation "
                             "baseline.")
    parser.add_argument("--method", type=str, default=None,
                        help="speculative method (dflash/dspark/eagle3); "
                             "auto-detected from the draft config if omitted")
    parser.add_argument("--num-spec-tokens", type=int, default=None,
                        help="speculative tokens per step; defaults to the "
                             "draft's block_size - 1 (dflash/dspark) or 3")
    parser.add_argument("--draft-sample-method", type=str,
                        default="probabilistic",
                        help="draft proposal sampling (probabilistic/greedy)")
    parser.add_argument("--dataset", type=str, default="yuekai/seed_tts_cosy2")
    parser.add_argument("--dataset-subset", type=str, default=None,
                        help="dataset config/subset name, if any")
    parser.add_argument("--dataset-split", type=str, default="test_zh")
    parser.add_argument("--prompt-tokens-key", type=str, default=None,
                        help="dataset column holding the prompt-audio speech "
                             "tokens; auto-detects a 'prompt_audio_*_tokens' "
                             "column if omitted")
    parser.add_argument("--no-prompt-tokens", action="store_true",
                        help="text-only prompting (skip the voice-clone "
                             "speech-token prefix)")
    parser.add_argument("--text-prefix", type=str, default=None,
                        help="prefix prepended to the user text; defaults to "
                             f"'{COSYVOICE3_PREFIX}' for CosyVoice3 targets "
                             "and '' otherwise")
    parser.add_argument("--batch-size", type=str, default="16",
                        help="comma-separated concurrent request counts, "
                             "e.g. '1,8,16,64'")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None,
                        help="only benchmark the first N samples")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output-json", type=str, default=None,
                        help="write results to this JSON file")
    return parser.parse_args()


def speech_id_str(tokens):
    return "".join(f"<|s_{t}|>" for t in tokens)


def load_draft_config(draft_model: str) -> dict:
    """Read the draft checkpoint's config.json (local path or HF hub)."""
    local = os.path.join(draft_model, "config.json")
    if os.path.isfile(local):
        with open(local) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download

    return json.load(open(hf_hub_download(draft_model, "config.json")))


def build_prompts(args, tokenizer):
    if os.path.isdir(args.dataset) and os.path.exists(
        os.path.join(args.dataset, "dataset_info.json")
    ):
        # local dataset directory produced by Dataset.save_to_disk()
        from datasets import load_from_disk

        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(
            args.dataset, args.dataset_subset, split=args.dataset_split
        )
    # LLM-only benchmark: drop audio columns so iteration doesn't decode audio
    # (which would require torchcodec/ffmpeg).
    drop = [
        c
        for c in dataset.column_names
        if c in ("prompt_audio", "target_audio", "audio")
    ]
    if drop:
        dataset = dataset.remove_columns(drop)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    tokens_key = args.prompt_tokens_key
    if tokens_key is None and not args.no_prompt_tokens:
        candidates = [
            c
            for c in dataset.column_names
            if c.startswith("prompt_audio") and c.endswith("_tokens")
        ]
        if not candidates:
            raise ValueError(
                f"No 'prompt_audio_*_tokens' column in {dataset.column_names}; "
                "pass --prompt-tokens-key or --no-prompt-tokens"
            )
        tokens_key = sorted(candidates)[-1]
        print(f"using prompt tokens column: {tokens_key}")

    prefix = args.text_prefix
    if prefix is None:
        is_cv3 = "cosyvoice3" in args.target_model.lower().replace("-", "")
        prefix = COSYVOICE3_PREFIX if is_cv3 else ""
        if prefix:
            print(f"auto text prefix: {prefix!r}")

    prompts = []
    for item in dataset:
        full_text = prefix + item["prompt_text"] + item["target_text"]
        for p in PUNCTS:
            full_text = full_text.replace(p, "")
        chat = [{"role": "user", "content": full_text}]
        if not args.no_prompt_tokens:
            chat.append(
                {"role": "assistant",
                 "content": speech_id_str(item[tokens_key])}
            )
        prompts.append(
            tokenizer.apply_chat_template(
                chat, tokenize=False, continue_final_message=True
            )
        )
    return prompts


def read_spec_counters(llm):
    counters = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name and hasattr(m, "value"):
                counters[m.name] = m.value
    except Exception:  # noqa: BLE001
        pass
    return counters


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompts = build_prompts(args, tokenizer)
    print(f"{len(prompts)} prompts from {args.dataset}:{args.dataset_split}")

    kwargs = dict(
        model=args.target_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_log_stats=False,
    )
    method = args.method
    num_spec = args.num_spec_tokens
    if args.draft_model:
        cfg = load_draft_config(args.draft_model)
        if method is None:
            method = cfg.get("speculators_model_type")
            if method is None:
                raise ValueError(
                    "Could not auto-detect the speculative method from the "
                    "draft config; pass --method"
                )
            print(f"auto method: {method}")
        if num_spec is None:
            block = cfg.get("block_size")
            num_spec = (block - 1) if block else 3
            print(f"auto num_spec_tokens: {num_spec}")
        kwargs["speculative_config"] = {
            "model": args.draft_model,
            "method": method,
            "num_speculative_tokens": num_spec,
            "draft_sample_method": args.draft_sample_method,
        }
        if args.repetition_penalty != 1.0:
            # Requires vllm-project/vllm#48932: mirror the target's
            # repetition penalty on the draft logits so the rejection test
            # compares aligned p/q distributions.
            kwargs["speculative_config"][
                "draft_apply_repetition_penalty"
            ] = True
            print("draft_apply_repetition_penalty: True (vllm PR #48932)")
    llm = LLM(**kwargs)

    def sp(seed):
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            seed=seed,
        )

    # warmup
    llm.generate(prompts[:4], sp(0), use_tqdm=False)

    results = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "method": method if args.draft_model else None,
        "num_spec_tokens": num_spec if args.draft_model else 0,
        "dataset": args.dataset,
        "split": args.dataset_split,
        "num_prompts": len(prompts),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "runs": {},
    }

    for bs in [int(b) for b in args.batch_size.split(",")]:
        spec_before = read_spec_counters(llm)
        t0 = time.perf_counter()
        out_tokens = 0
        finished_by_stop = 0
        for i in range(0, len(prompts), bs):
            chunk = prompts[i : i + bs]
            outs = llm.generate(
                chunk,
                [sp(10_000 * bs + i + j) for j in range(len(chunk))],
                use_tqdm=False,
            )
            for o in outs:
                gen = o.outputs[0]
                out_tokens += len(gen.token_ids)
                finished_by_stop += gen.finish_reason == "stop"
        wall = time.perf_counter() - t0
        spec_after = read_spec_counters(llm)

        run = {
            "wall_s": round(wall, 2),
            "output_tokens": out_tokens,
            "tok_per_s": round(out_tokens / wall, 1),
            # 25 speech tokens = 1s of audio
            "audio_seconds_generated": round(out_tokens / 25, 1),
            "audio_realtime_factor": round(out_tokens / 25 / wall, 2),
            "finished_by_stop": finished_by_stop,
        }
        drafts = spec_after.get(
            "vllm:spec_decode_num_drafts", 0
        ) - spec_before.get("vllm:spec_decode_num_drafts", 0)
        accepted = spec_after.get(
            "vllm:spec_decode_num_accepted_tokens", 0
        ) - spec_before.get("vllm:spec_decode_num_accepted_tokens", 0)
        draft_toks = spec_after.get(
            "vllm:spec_decode_num_draft_tokens", 0
        ) - spec_before.get("vllm:spec_decode_num_draft_tokens", 0)
        if drafts:
            run["spec"] = {
                "acceptance_rate": round(accepted / draft_toks, 4),
                "mean_acceptance_len": round(1 + accepted / drafts, 3),
            }
        results["runs"][f"bs{bs}"] = run
        print(f"bs={bs}: {json.dumps(run)}")

    print(json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"results written to {args.output_json}")


if __name__ == "__main__":
    main()
