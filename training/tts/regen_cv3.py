"""On-policy regeneration for CosyVoice3: the verifier samples its own speech
tokens for Emilia-ZH texts, with the CV3 prompt prefix
'You are a helpful assistant.<|endofprompt|>'.

Only samples that stop naturally at <|eos1|> are kept.
"""

import json
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF"
PREFIX = "You are a helpful assistant.<|endofprompt|>"

SRC = sys.argv[1]
DST = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(MODEL)

items = [json.loads(line) for line in open(SRC)]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": PREFIX + it["text"]}], tokenize=False
    )
    for it in items
]

llm = LLM(model=MODEL, gpu_memory_utilization=0.8, max_model_len=4096)
# CV3 official sampling params (matches deployment; rp1.0/top_k50 caused
# runaway generation and a draft-invisible distribution mismatch)
sp = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=15,
    repetition_penalty=1.1,
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
        conv = {
            "id": it["id"],
            "conversations": [
                {"role": "user", "content": PREFIX + it["text"]},
                {"role": "assistant", "content": gen.text + "<|eos1|>"},
            ],
        }
        fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
        kept += 1

print(f"kept {kept}, dropped (length-truncated) {dropped}")
