# Training the speculative drafters

Two separate stacks, one shared principle: **train on target-regenerated
(on-policy) labels**, not ground-truth annotations. The draft has to predict
what the *target model* emits under deployment decoding — human labels differ
in casing/punctuation (ASR) or token choice under sampling (TTS), and that
train/serve mismatch costs most of the acceptance. On-policy regeneration was
the single biggest lever in both stacks (+25% TTS throughput; the ASR drafts
were trained this way from the start).

## 1. Speech LLM drafts (EAGLE3 / DFlash for Qwen2-Audio & Qwen3-Omni) — SpecForge

Trained with the SpecForge fork:
**<https://github.com/yuekaizhang/SpecForge/tree/feat/qwen2-audio-eagle3>**
(adds audio-target support: Qwen2-Audio and Qwen3-Omni thinker targets, omni
mel collators, nested-`text_config` LM-head tie detection, DFlash for
composite models, and the transformers-5.x `rope_parameters` fix).

Pipeline (produces the four checkpoints in
[../README_SPEECH_LLM.md](../README_SPEECH_LLM.md)):

1. **Regenerate labels with the target** — serve the target with sglang and
   transcribe the training audio (greedy, the same prompt used at eval):

   ```bash
   # 4 servers x tp=2 on 8xH100, then:
   python scripts/generate_response.py \
       --dataset openslr/librispeech_asr --subset clean --split train.100 \
       --prompt "Transcribe the English audio into text." \
       --output labels.jsonl        # {"idx", "id", "gt", "target_gen"}
   ```

   For the multi-domain omni draft this was done over 6 English ASR corpora
   (~500k utts: GigaSpeech-s, AMI-ihm, SPGISpeech-S, Earnings22, LibriSpeech
   train.100, VoxPopuli-en), assembled with `scripts/assemble_multidomain.py`.
   The assembled dataset (audio + regenerated transcription) is published at
   [yuekai/multi_en_qwen3_omni_sglang_regenerated](https://huggingface.co/datasets/yuekai/multi_en_qwen3_omni_sglang_regenerated).

2. **Train the draft** — example launchers in SpecForge `examples/`:

   | Draft | Launcher | Key hparams |
   |---|---|---|
   | Qwen2-Audio EAGLE3 | `run_qwen2_audio_eagle3.sh` | ttt, lr 1e-5 |
   | Qwen2-Audio DFlash block-4 | `run_qwen2_audio_dflash.sh` | `--lm-head-key language_model.lm_head.weight` (untied head!) |
   | Qwen3-Omni EAGLE3 (audio) | `run_qwen3_omni_eagle3_multidomain.sh` | ttt7, 8xH100, FSDP must `ignored_modules=[target]` |
   | Qwen3-Omni DFlash block-8 (mix) | `run_qwen3_omni_dflash_text.sh` + multidomain data | lr 6e-4, warmup 0.04, `--loss-decay-gamma`, block 8 |

   DFlash defaults follow the paper recipe (AdamW wd=0, cosine to 0, lr 6e-4,
   warmup ratio 0.04); the audio-only runs that used lr 1e-5 converged far
   slower — prefer 6e-4.

3. **Serve & verify** — [../benchmark_speech_llm.py](../benchmark_speech_llm.py)
   end to end (sglang fork `yuekaizhang/sglang@speculative`).

Gotchas worth knowing (each cost a debugging session):

- **Composite-config LM-head tying**: read `text_config.tie_word_embeddings`,
  not the top-level flag — Qwen2-Audio ships a real untied head under a
  top-level `tie_word_embeddings=True`. Training against the wrong head makes
  the draft unusable (accept 1.0) and is only detectable at serve time.
- **transformers 5.x rope**: `rope_theta` moved into `config.rope_parameters`;
  a bare `getattr(config, "rope_theta", 10000)` trains at base 10000 while
  sglang serves 1e6. Worth +0.2–0.6 accept once matched.
- **Serve DFlash at its training block size** — shrinking
  `--speculative-num-draft-tokens` below the trained block only drops accept.

## 2. TTS drafters (DSpark / DFlash for CosyVoice2/3) — speculators

Trained with the speculators fork:
**<https://github.com/yuekaizhang/speculators/tree/tts>**
(`scripts/train.py`; needs `>= 69150da` for qwen2-style verifiers that omit
`head_dim`). Scripts in [tts/](tts/):

| Script | Produces | Notes |
|---|---|---|
| [tts/regen_cv2.py](tts/regen_cv2.py) | on-policy CV2 labels | regenerates Emilia speech-token continuations with `yuekai/cosyvoice2_llm` under deployment sampling (temp 0.8 / top_p 0.95 / top_k 50) |
| [tts/regen_cv3.py](tts/regen_cv3.py) | on-policy CV3 labels | CV3 official params (top_k 15, **rep_penalty 1.1** — mandatory, CV3 runs away without it) |
| [tts/train_dspark_cv2.sh](tts/train_dspark_cv2.sh) | `yuekai/cosyvoice2_llm_dspark` | 3 layers, block 8, 8192 draft vocab — recommended CV2 drafter (2.1–3.1x) |
| [tts/train_dflash_cv2.sh](tts/train_dflash_cv2.sh) | `yuekai/cosyvoice2_llm_dflash` | 5 layers, block 4 — best large-batch throughput |
| [tts/train_dspark_cv3.sh](tts/train_dspark_cv3.sh) | `yuekai/cosyvoice3_llm_dspark` | pair with vLLM `draft_apply_repetition_penalty` ([vllm#48932](https://github.com/vllm-project/vllm/pull/48932)) |

Empirical notes from the CV2/CV3 campaign (full log: the experiments table in
the TTS work dir):

- **Parallel drafters only**: an autoregressive EAGLE3 draft on a 0.5B target
  costs more than it saves; DSpark/DFlash (one draft forward per block) win.
- **`draft_sample_method="probabilistic"`** (vLLM >= 0.21) is a prerequisite —
  the default greedy-proposal acceptance is capped by the target's top-1
  probability (~0.25 for TTS sampling) and speculation always loses.
- **Draft capacity doesn't buy acceptance** — 5-layer DSpark ties 3-layer on
  accept (3.27 vs 3.26) and loses end to end on draft cost. Data (on-policy,
  more of it) is what moves accept.
- 8192-entry reduced draft vocab (all speech tokens + eos + frequent text via
  d2t mapping) is equivalent in accept to the full 158k vocab and much faster.
