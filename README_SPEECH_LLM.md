# Speculative decoding for speech LLMs (ASR)

Lossless speculative decoding for audio-input LLMs served with sglang:
EAGLE3 and [DFlash](https://github.com/z-lab/dflash) drafts for
**Qwen3-Omni-30B-A3B-Instruct** (thinker, text output) and
**Qwen2-Audio-7B AISHELL SFT**, giving **1.5–1.65x** end-to-end ASR speedups
at batch=1 with identical WER/CER.

## Models

| HF repo | What it is |
|---|---|
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Qwen3-Omni target (48-layer A3B MoE thinker) |
| [yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio) | EAGLE3 draft (ttt7), trained on 500k multi-domain English ASR utterances |
| [yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8](https://huggingface.co/yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8) | DFlash draft (5 layers, block 8), trained on a text (open-perfectblend) + audio (multi-domain ASR) mix |
| [yuekai/qwen2_audio_aishell_sft](https://huggingface.co/yuekai/qwen2_audio_aishell_sft) | Qwen2-Audio-7B target, SFT on AISHELL-1 |
| [yuekai/qwen2_audio_aishell_sft_eagle3](https://huggingface.co/yuekai/qwen2_audio_aishell_sft_eagle3) | EAGLE3 draft, trained on AISHELL-1 |
| [yuekai/qwen2_audio_aishell_sft_dflash_block4](https://huggingface.co/yuekai/qwen2_audio_aishell_sft_dflash_block4) | DFlash draft (5 layers, block 4), trained on AISHELL-1 |

All drafts are trained with SpecForge on target-regenerated labels (the target
model transcribes the training audio itself, so the draft learns the target's
output distribution — casing, punctuation — not the human annotation style).
See [training/README.md](training/README.md).

## Requirements (exact versions for reproduction)

Speech-LLM speculative decoding (EAGLE3 + DFlash shims for Qwen2-Audio and
Qwen3-Omni) is not yet in upstream sglang. Use this fork:

```bash
git clone -b speculative https://github.com/yuekaizhang/sglang
# benchmarked at commit 00e1106f4ab55cfb7dceeba2dfbc04b0cc40bb6f
cd sglang && pip install -e python
```

Benchmarked with: `transformers==5.8.1`, `torch==2.11`, 1×H100, CUDA graphs on
(`--cuda-graph-max-bs-decode 1` for the batch=1 numbers below).

## Usage

`benchmark_speech_llm.py` is end-to-end: it launches the sglang server itself
(baseline or speculative — the algorithm and block size are inferred from the
draft's `config.json`), decodes the full test set, and reports WER/CER,
latency, throughput, and accept length. Datasets: `librispeech`
(test-clean, 2620 utts, WER) or `aishell` (test, 7176 utts, CER).

```bash
# Qwen3-Omni on LibriSpeech test-clean: baseline / EAGLE3 / DFlash
python3 benchmark_speech_llm.py --target-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --dataset librispeech --output-dir results/omni_baseline
python3 benchmark_speech_llm.py --target-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --draft-model yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio \
    --dataset librispeech --output-dir results/omni_eagle3
python3 benchmark_speech_llm.py --target-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --draft-model yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8 \
    --dataset librispeech --output-dir results/omni_dflash

# Qwen2-Audio AISHELL SFT on AISHELL test: baseline / EAGLE3 / DFlash
python3 benchmark_speech_llm.py --target-model yuekai/qwen2_audio_aishell_sft \
    --dataset aishell --output-dir results/q2a_baseline
python3 benchmark_speech_llm.py --target-model yuekai/qwen2_audio_aishell_sft \
    --draft-model yuekai/qwen2_audio_aishell_sft_eagle3 \
    --dataset aishell --output-dir results/q2a_eagle3
python3 benchmark_speech_llm.py --target-model yuekai/qwen2_audio_aishell_sft \
    --draft-model yuekai/qwen2_audio_aishell_sft_dflash_block4 \
    --dataset aishell --output-dir results/q2a_dflash
```

Useful flags: `--num-samples N` (quick smoke), `--concurrency C`,
`--tp N`, `--server URL` (reuse a running server),
`--num-steps/--eagle-topk/--num-draft-tokens` (EAGLE3 decode params;
DFlash must be served at its training block size — the default).

## Results (1×H100, tp=1, batch=1, greedy, full test sets)

### Qwen3-Omni-30B-A3B-Instruct — LibriSpeech test-clean (2620 utts)

| config | WER | accept | mean lat | wall | utt/s | speedup |
|---|---|---|---|---|---|---|
| baseline (no draft) | 1.70% | – | 0.195 s | 514 s | 5.1 | 1.00x |
| + EAGLE3 audio (steps 3, topk 1, draft 4) | 1.68% | 3.12 | 0.131 s | 350.3 s | 7.5 | 1.49x |
| + DFlash block-8 (text+audio mix) | 1.70% | 4.23 | 0.124 s | 330.4 s | 7.9 | **1.57x** |

The DFlash mix draft also speeds up text tasks (2.14x on gsm8k, 1.11x on
alpaca) because half its training data is text — the EAGLE3 draft is
audio-only and was not evaluated on text.

### Qwen2-Audio-7B AISHELL SFT — AISHELL-1 test (7176 utts)

| config | CER | accept | mean lat | wall | utt/s | speedup |
|---|---|---|---|---|---|---|
| baseline (no draft) | 2.01% | – | 0.131 s | 944.1 s | 7.6 | 1.00x |
| + EAGLE3 (steps 3, topk 1, draft 4) | 2.00% | 3.37 | 0.079 s | 573.2 s | 12.5 | **1.65x** |
| + DFlash block-4 | 2.01% | 3.29 | 0.082 s | 596.4 s | 12.0 | 1.58x |

### Reproduction check (2026-08-03)

All six configs were re-run end to end with `benchmark_speech_llm.py`, pulling
the four draft checkpoints straight from the HF hub (sglang fork @ `00e1106f4a`,
sgl-kernel 0.4.5). Quality and accept lengths reproduce:

| run | quality | accept (micro) | original |
|---|---|---|---|
| omni baseline | WER 1.70% | – | 1.70% |
| omni + eagle3_audio | WER 1.68% | 3.33 | 3.12 |
| omni + dflash_block8 | WER 1.70% | 4.77 | 4.23 |
| q2a baseline | CER 1.95% | – | 2.01% |
| q2a + eagle3 | CER 1.95% | 3.37 | 3.37 |
| q2a + dflash_block4 | CER 1.95% | 3.29 | 3.29 |

The wall-time/speedup columns in the tables above are from the original
single-family campaign (the repro ran 6 servers on one node concurrently, which
contends CPU and distorts absolute wall time; accept and WER/CER are unaffected).
Per-run summaries: `results/repro/*/summary.txt`.

Notes:

- All configs are lossless (greedy verification): WER/CER matches the baseline.
- Accept length = completion_tokens / spec_verify_ct (tokens per target decode
  forward, includes the bonus token).
- ASR drafts are domain-sensitive: these numbers hold in-domain; expect accept
  to drop on far-out-of-domain audio (see the training README for the
  multi-domain data recipe that mitigates this).
