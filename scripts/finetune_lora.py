"""LoRA fine-tune for Gauge's local writer model, using Unsloth (NVIDIA GPU, Linux or WSL2).

    python scripts/finetune_lora.py --datasets data/models/datasets --base unsloth/llama-3.1-8b-instruct --out data/models/lora

Trains SFT on sft.jsonl, then ORPO on the preference files (self_preference, author_edits,
preference), exports GGUF, and prints the `ollama create` command. Everything here is standard
Unsloth / TRL usage; adjust epochs and ranks for your card. Not run by tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8")] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--base", default="unsloth/llama-3.1-8b-instruct")
    ap.add_argument("--out", default="data/models/lora")
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    from datasets import Dataset  # type: ignore
    from trl import ORPOConfig, ORPOTrainer, SFTConfig, SFTTrainer  # type: ignore
    from unsloth import FastLanguageModel  # type: ignore

    ds = Path(args.datasets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = FastLanguageModel.from_pretrained(args.base, max_seq_length=args.max_seq, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    sft_rows = load_jsonl(ds / "sft.jsonl")
    if sft_rows:
        texts = [f"### Instruction:\n{r['instruction']}\n\n### Input:\n{r['input']}\n\n### Response:\n{r['output']}" for r in sft_rows]
        SFTTrainer(model=model, tokenizer=tok, train_dataset=Dataset.from_dict({"text": texts}),
                   args=SFTConfig(output_dir=str(out / "sft"), num_train_epochs=args.epochs, per_device_train_batch_size=1,
                                  gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=5, dataset_text_field="text",
                                  max_seq_length=args.max_seq)).train()

    pref = load_jsonl(ds / "self_preference.jsonl") + load_jsonl(ds / "author_edits.jsonl") + load_jsonl(ds / "preference.jsonl")
    if pref:
        ORPOTrainer(model=model, tokenizer=tok,
                    train_dataset=Dataset.from_dict({"prompt": [r["prompt"] for r in pref], "chosen": [r["chosen"] for r in pref],
                                                     "rejected": [r["rejected"] for r in pref]}),
                    args=ORPOConfig(output_dir=str(out / "orpo"), num_train_epochs=args.epochs, per_device_train_batch_size=1,
                                    gradient_accumulation_steps=8, learning_rate=8e-6, logging_steps=5, max_length=args.max_seq)).train()

    model.save_pretrained_gguf(str(out / "gguf"), tok, quantization_method="q4_k_m")
    (out / "Modelfile").write_text(f"FROM {out / 'gguf'}\nPARAMETER num_ctx 8192\n", encoding="utf-8")
    print(f"\nDone. Register with Ollama:\n  ollama create gauge-writer -f {out / 'Modelfile'}\nthen: gauge init --model gauge-writer")


if __name__ == "__main__":
    main()
