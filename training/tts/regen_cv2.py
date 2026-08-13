"""On-policy regeneration: replace ground-truth codes with the verifier's own
sampled speech tokens, using the exact deployment sampling params.

Only samples that stop naturally at <|eos1|> are kept (hitting max_tokens would
teach the draft a wrong eos distribution).
"""

import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

BASE = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/outputs/cosyvoice2_eagle3"
MODEL = f"{BASE}/cosyvoice2_llm_nobias"
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/cosy_v2_tokens_ZH_10k.jsonl"
DST = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/train_conversations_10k_regen.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

items = [json.loads(line) for line in open(SRC)]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": it["text"]}], tokenize=False
    )
    for it in items
]

llm = LLM(model=MODEL, gpu_memory_utilization=0.8, max_model_len=4096)
# deployment sampling params (rep penalty dropped per deployment decision)
sp = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.0,
    max_tokens=2048,
)
outputs = llm.generate(prompts, sp)

kept = dropped = 0
with open(DST, "w") as fout:
    for it, out in zip(items, outputs):
        gen = out.outputs[0]
        if gen.finish_reason != "stop":
            dropped += 1
            continue
        # generated text is the speech token string; eos itself is not included
        speech = gen.text + "<|eos1|>"
        conv = {
            "id": it["id"],
            "conversations": [
                {"role": "user", "content": it["text"]},
                {"role": "assistant", "content": speech},
            ],
        }
        fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
        kept += 1

print(f"kept {kept}, dropped (length-truncated) {dropped}")
