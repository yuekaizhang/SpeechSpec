# SpeechSpec

Lossless speculative decoding for speech models — trained drafters, serving
recipes, and end-to-end benchmarks.

| Part | Doc | TL;DR |
|---|---|---|
| **Speech LLM (ASR)** | [README_SPEECH_LLM.md](README_SPEECH_LLM.md) | EAGLE3 / DFlash drafts for Qwen3-Omni-30B and Qwen2-Audio in **sglang**: 1.5–1.65x at batch=1, WER/CER unchanged. `benchmark_speech_llm.py` reproduces everything end to end. |
| **TTS (text → speech tokens)** | [README_TTS.md](README_TTS.md) | DSpark / DFlash drafters for CosyVoice2/3 LLMs in **vLLM**: 1.2–3.1x across batch sizes. `benchmark_tts.py`. |
| **Training** | [training/README.md](training/README.md) | How the drafts above were trained: SpecForge (speech LLM) and vllm-project/speculators (TTS), both on target-regenerated (on-policy) labels. |

All draft checkpoints are on the HF hub under [yuekai](https://huggingface.co/yuekai);
each doc lists its exact model/dataset repos, framework versions, and
measured numbers.
