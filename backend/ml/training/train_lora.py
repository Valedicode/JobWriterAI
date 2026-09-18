"""LoRA / QLoRA SFT: (resume_text, job_description_text) -> teacher compatibility JSON.

Run from backend/:
  Smoke test (CPU OK, ~1 min, no teacher labels needed):
    python -m ml.training.train_lora --smoke
  Real run (Kaggle GPU; QLoRA auto-enabled when CUDA + bitsandbytes are present):
    python -m ml.training.train_lora --labels data/ml/teacher_labels/full.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
On Kaggle: upload backend/ as a dataset, `pip install -r ml/requirements-ml.txt`, then run the line above.
"""
import argparse
import json
import math
import random

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from ml.config import CHECKPOINTS, HF_CACHE, HF_DATASET, SEED

SYSTEM = "Score how well the resume fits the job. Answer with JSON only."
DIMENSIONS = ("hard_skills", "experience", "seniority", "domain", "ats_keywords")
SMOKE_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def target_json(report):
    dims = {d["name"]: round(d["score"], 2) for d in report["dimensions"]}
    return json.dumps({"aggregate_score": round(report["aggregate_score"], 2), "level": report["level"],
                       "dimensions": {k: dims[k] for k in DIMENSIONS}})


def build_example(tok, resume, job, report, resume_tokens, job_tokens):
    # Truncate inputs, never the completion: trainer-side truncation would cut the JSON target.
    cut = lambda text, n: tok.decode(tok(text, add_special_tokens=False)["input_ids"][:n])
    user = f"Resume:\n{cut(resume, resume_tokens)}\n\nJob description:\n{cut(job, job_tokens)}"
    return {"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": target_json(report)}]}


def toy_rows(n):
    """Real HF text with random fake reports: exercises formatting/truncation/masking, not quality."""
    rng = random.Random(SEED)
    ds = load_dataset(HF_DATASET, cache_dir=str(HF_CACHE))["train"].shuffle(seed=SEED).select(range(n))
    for r in ds:
        agg = rng.random()
        yield r["resume_text"], r["job_description_text"], {
            "aggregate_score": agg, "level": rng.choice(["low", "medium", "high", "excellent"]),
            "dimensions": [{"name": d, "score": rng.random()} for d in DIMENSIONS]}


def teacher_rows(path):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["status"] == "ok" and not r["flags"]:
            yield r["resume_text"], r["job_description_text"], r["report"]


def load_model(name):
    quantize = torch.cuda.is_available()
    kwargs = {}
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            "device_map": "auto"}
    return AutoModelForCausalLM.from_pretrained(name, **kwargs), quantize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--labels")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    # ~p90 of HF train lengths in Qwen tokens (data/ml/hf_eval_cache/dataset_report.json)
    ap.add_argument("--resume-tokens", type=int, default=1536)
    ap.add_argument("--job-tokens", type=int, default=768)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    if args.smoke:
        args.model, args.resume_tokens, args.job_tokens = SMOKE_MODEL, 128, 96

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = toy_rows(16) if args.smoke else teacher_rows(args.labels)
    rows = list(rows)
    # Group validation by job: the HF data has ~22 rows per job, so a row-level split leaks jobs.
    jobs = sorted({job for _, job, _ in rows})
    random.Random(SEED).shuffle(jobs)
    val_jobs = set(jobs[: max(1, round(len(jobs) * (0.25 if args.smoke else args.val_frac)))])
    make = lambda keep: Dataset.from_list([build_example(tok, *r, args.resume_tokens, args.job_tokens)
                                           for r in rows if (r[1] in val_jobs) == keep])
    split = {"train": make(False), "test": make(True)}
    print(f"train {len(split['train'])} rows / val {len(split['test'])} rows ({len(val_jobs)} held-out jobs)")

    model, quantized = load_model(args.model)
    bf16 = quantized and torch.cuda.is_bf16_supported()
    out_dir = CHECKPOINTS / ("smoke" if args.smoke else args.model.split("/")[-1])
    config = SFTConfig(
        output_dir=str(out_dir), seed=SEED,
        max_length=args.resume_tokens + args.job_tokens + 256,  # + chat template + JSON target
        completion_only_loss=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1 if args.smoke else 8,
        num_train_epochs=args.epochs, max_steps=12 if args.smoke else -1,
        learning_rate=1e-2 if args.smoke else 2e-4,  # smoke: high LR so a falling loss proves LoRA learns lr_scheduler_type="cosine", warmup_steps=0.03,  # float = ratio in transformers 5
        gradient_checkpointing=quantized, bf16=bf16, fp16=quantized and not bf16,
        eval_strategy="epoch" if not args.smoke else "no", save_strategy="epoch" if not args.smoke else "no",
        logging_steps=1 if args.smoke else 10, report_to="none",
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=model, args=config, train_dataset=split["train"], eval_dataset=split["test"],
                         processing_class=tok, peft_config=lora)

    if args.smoke:
        smoke_checks(trainer, tok)
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    if args.smoke:
        smoke_after_train(trainer, tok, split["test"][0], out_dir / "adapter")


def smoke_checks(trainer, tok):
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    supervised = tok.decode(labels[labels != -100])
    assert "aggregate_score" in supervised, f"target missing from loss: {supervised!r}"
    assert "Resume:" not in supervised, "prompt tokens are not masked out of the loss"
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    assert 0 < trainable < sum(p.numel() for p in trainer.model.parameters()), "LoRA not applied"
    print(f"[smoke] masking OK, supervised span: {supervised[:120]!r}; trainable params: {trainable}")


def smoke_after_train(trainer, tok, example, adapter_dir):
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    assert losses and all(math.isfinite(l) for l in losses), f"bad losses {losses}"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir))
    inputs = tok.apply_chat_template(example["prompt"], add_generation_prompt=True,
                                     return_dict=True, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    print(f"[smoke] losses {[round(l, 3) for l in losses]}; adapter reload + generate OK "
          f"(tiny random model, output is gibberish by design): {ascii(tok.decode(new_tokens))}")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
