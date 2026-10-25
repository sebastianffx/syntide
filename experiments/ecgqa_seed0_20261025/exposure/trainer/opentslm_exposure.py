"""Fixed original-QA-exposure ablation derived from the archived CUDA trainer.

Every update keeps the baseline's two original batches of eight and adds a
separate generated batch of two. Stop at 12,000 updates while retaining the
baseline's 15,531-update learning-rate horizon. Selection is diagnostic only.

SYNTIDE owns the input manifest and metrics; this module owns a conventional
PyTorch loop. No model, tokenizer, or CUDA smoke is run on the Mac. Example::

    python -m syntide_trainers.opentslm_loop train --prepared data --run-dir runs/seed0 \
        --seed 0 --max-steps 8000 --batch-size 8 --val-every 500
    python -m syntide_trainers.opentslm_loop evaluate --prepared data --run-dir runs/seed0 \
        --split validation --batch-size 32

The official upstream model is used, with a frozen base Gemma checkpoint. The
collator fixes two upstream bugs: exactly one media marker for the one joint
12-lead media item, and answer-only labels counted including BOS/padding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

import exposure_protocol as exposure

UPSTREAM_COMMIT = "2968f4b891baab4307f7e9d0043e87677b593a30"
BASE_MODEL = "google/gemma-3-270m"
PROMPT_VERSION = "ecgqa-cot-single-joint-media-v1"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def write_json(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return f"source@{UPSTREAM_COMMIT}" if package == "opentslm" else "not installed as a distribution"


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This baseline runs only on RunPod CUDA; CPU and MPS training are disabled.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This baseline requires a CUDA GPU with native bfloat16 support.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


class Prepared:
    """Only open the requested QA split; training never reads test questions."""

    def __init__(self, path: Path, splits: tuple[str, ...]):
        from syntide.baseline.data import verify_prepared

        self.path = path
        self.manifest = verify_prepared(path, splits=splits)
        self.manifest_sha256 = self.manifest["sha256"]
        self.records = json.loads((path / "records.json").read_text())
        self.record_index = {record: i for i, record in enumerate(self.records)}
        self.signals = np.load(path / "signals.npy", mmap_mode="r", allow_pickle=False)
        if self.signals.shape != (len(self.records), 12, 1000):
            raise ValueError(f"Expected (N,12,1000) signals; received {self.signals.shape}")
        self.rows = {split: pq.read_table(path / f"{split}.parquet").to_pylist() for split in splits}
        self.file_sha256 = {
            filename: self.manifest["files"][filename]["sha256"]
            for filename in ("records.json", "signals.npy", *(f"{split}.parquet" for split in splits))
        }
        for split, rows in self.rows.items():
            if not rows:
                raise ValueError(f"Empty {split} split")
            if any(row["record_id"] not in self.record_index for row in rows):
                raise ValueError(f"Unknown record in {split}")
        if {"train", "selection"}.issubset(splits):
            train_patients = {str(row["subject_uid"]) for row in self.rows["train"]}
            selection_patients = {str(row["subject_uid"]) for row in self.rows["selection"]}
            if train_patients & selection_patients:
                raise ValueError("Patient leakage between training and checkpoint selection")

    def signal(self, row: dict, zero: bool = False) -> np.ndarray:
        if zero:
            return np.zeros((12, 1000), dtype=np.float32)
        return self.signals[self.record_index[row["record_id"]]]


def make_prompt(row: dict, signal: np.ndarray) -> tuple[str, np.ndarray]:
    """Upstream ECG wording/statistics and per-lead z scores; joint ECG marker."""
    signal = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite ECG: {row['record_id']}")
    means = signal.mean(axis=-1, keepdims=True)
    stds = signal.std(axis=-1, keepdims=True)
    normalized = (signal - means) / np.where(stds > 1e-6, stds, 1.0)
    lead_descriptions = "\n".join(
        f"This is ECG Lead {name}, it has mean {float(means[i, 0]):.4f} and std {float(stds[i, 0]):.4f}:"
        for i, name in enumerate(LEADS)
    )
    options = ", ".join(str(option) for option in row["answer_options"])
    paragraph_instruction = (
        "- Write your rationale as a single, natural paragraph — "
        "do not use bullet points, numbered steps, or section headings."
    )
    text = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context: {row["clinical_context"]}

Your task is to examine the ECG signal and answer the following medical question:

Question: {row["question"]}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
{paragraph_instruction}
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question.

<image> {lead_descriptions} <|endofchunk|>

Based on your analysis of the ECG data, select your answer from the following options:
{options}

- Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""
    return text, np.ascontiguousarray(normalized)


def target_text(row: dict) -> str:
    # The short gold answer is authoritative; remove a duplicated final Answer
    # from the source rationale before appending it exactly once.
    rationale = re.split(r"(?i)\banswer\s*:", row["rationale"])[0].strip()
    return f"{rationale}\nAnswer: {row['answer']}".strip()


class Collator:
    def __init__(self, tokenizer, prepared: Prepared, max_length: int):
        self.tokenizer = tokenizer
        self.prepared = prepared
        self.max_length = max_length
        self.prompt_cache: dict[str, tuple[str, np.ndarray]] = {}
        self.signal_cache: dict[str, np.ndarray] = {}
        self.training_token_cache: dict[str, tuple[list[int], list[int]]] = {}
        self.truncated_tasks: set[str] = set()

    def prompt(self, row: dict, zero: bool = False) -> tuple[str, np.ndarray]:
        key = row["task_id"]
        if key not in self.prompt_cache:
            prompt, normalized = make_prompt(row, self.prepared.signal(row))
            normalized = self.signal_cache.setdefault(row["record_id"], normalized)
            self.prompt_cache[key] = (prompt, normalized)
        prompt, normalized = self.prompt_cache[key]
        return prompt, np.zeros_like(normalized) if zero else normalized

    def encode(self, row: dict, *, training: bool, zero: bool = False, option: str | None = None):
        prompt, signal = self.prompt(row, zero=zero)
        tokenizer = self.tokenizer
        if option is not None:
            # Fixed answer-only option likelihood, used only when the CUDA smoke
            # shows rationale generation is too expensive for checkpoint selection.
            prompt += "\nAnswer:"
        cache_key = row["task_id"]
        if training and not zero and option is None and cache_key in self.training_token_cache:
            ids, labels = self.training_token_cache[cache_key]
            return ids, labels, signal
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if training:
            target = " " + (target_text(row) if option is None else option)
            target_ids = [*tokenizer.encode(target, add_special_tokens=False), tokenizer.eos_token_id]
            # Truncate rationale, preserving the final answer and EOS. Never
            # truncate the question, the ECG statistics, or the allowed options.
            if len(prompt_ids) + len(target_ids) > self.max_length:
                room = self.max_length - len(prompt_ids)
                answer_ids = tokenizer.encode(f"\nAnswer: {row['answer']}", add_special_tokens=False)
                if room < len(answer_ids) + 2 or option is not None:
                    raise ValueError(f"Prompt exceeds max length for {row['task_id']}; increase --max-length")
                target_ids = target_ids[: room - len(answer_ids) - 1] + answer_ids + [tokenizer.eos_token_id]
                self.truncated_tasks.add(cache_key)
            ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            if option is None and not zero:
                self.training_token_cache[cache_key] = (ids, labels)
        else:
            if len(prompt_ids) > self.max_length:
                raise ValueError(f"Prompt exceeds max length for {row['task_id']}; increase --max-length")
            ids, labels = prompt_ids, None
        if ids.count(tokenizer.convert_tokens_to_ids("<image>")) != 1:
            raise ValueError("Expected exactly one media token for the joint 12-lead ECG")
        return ids, labels, signal

    def __call__(self, rows: list[dict], *, training: bool, zero: bool = False, options=None) -> dict:
        encoded = [
            self.encode(row, training=training, zero=zero, option=None if options is None else options[i])
            for i, row in enumerate(rows)
        ]
        length = max(len(item[0]) for item in encoded)
        input_ids = torch.full((len(rows), length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100) if training else None
        for i, (ids, targets, _) in enumerate(encoded):
            # Right padding during loss; left padding during autoregressive decoding.
            start = 0 if training else length - len(ids)
            input_ids[i, start : start + len(ids)] = torch.tensor(ids)
            attention_mask[i, start : start + len(ids)] = 1
            if training:
                labels[i, start : start + len(ids)] = torch.tensor(targets)
        batch = {
            "lang_x": input_ids.cuda(non_blocking=True),
            "attention_mask": attention_mask.cuda(non_blocking=True),
            "vision_x": torch.from_numpy(np.stack([item[2] for item in encoded])).unsqueeze(1).cuda(),
        }
        if training:
            batch["labels"] = labels.cuda(non_blocking=True)
        return batch


def token_length_summary(collator: Collator, rows: list[dict], sample_size: int = 1024) -> dict:
    """Inspect the text lengths on CUDA host before spending training time."""
    chosen = rows[:: max(1, len(rows) // sample_size)][:sample_size]
    prompt_lengths, target_lengths = [], []
    for row in chosen:
        prompt, _ = collator.prompt(row)
        prompt_lengths.append(len(collator.tokenizer.encode(prompt, add_special_tokens=True)))
        target_lengths.append(
            len(collator.tokenizer.encode(" " + target_text(row), add_special_tokens=False)) + 1
        )
    totals = np.array(prompt_lengths) + np.array(target_lengths)
    return {
        "sample_questions": len(chosen),
        "max_prompt_tokens": max(prompt_lengths),
        "target_tokens_p50_p90_p99_max": np.percentile(target_lengths, [50, 90, 99, 100]).tolist(),
        "total_tokens_p50_p90_p99_max": np.percentile(totals, [50, 90, 99, 100]).tolist(),
        "would_truncate_rationale": int((totals > collator.max_length).sum()),
    }


def answer_losses(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact teacher-forced NLL, projecting only tokens supervised by the loss.

    Gemma has a 262k-word vocabulary. Skipping the masked prompt logits avoids
    their substantial allocation. The upstream encoder, Perceiver, Flamingo
    decoder, and next-token objective are otherwise unchanged.
    """
    core = model.model
    shifted_labels = batch["labels"][:, 1:]
    positions = (shifted_labels != -100).any(dim=0).nonzero(as_tuple=True)[0]
    core._encode_vision_x(batch["vision_x"])
    try:
        output = core.lang_encoder(
            input_ids=batch["lang_x"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            logits_to_keep=positions,
        )
        labels = shifted_labels[:, positions]
        losses = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(labels.shape)
        return losses, labels != -100
    finally:
        core.lang_encoder.clear_conditioned_layers()


def upstream_source_commit() -> str:
    import opentslm

    source_root = Path(opentslm.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"OpenTSLM must be pinned at {UPSTREAM_COMMIT}, found {commit}")
    subprocess.run(
        ["git", "-C", str(source_root), "diff", "--exit-code", "HEAD", "--", "src"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return commit


def load_model(seed: int, revision: str):
    from huggingface_hub import snapshot_download
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

    upstream_source_commit()
    snapshot = Path(
        snapshot_download(
            BASE_MODEL,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "special_tokens*"],
        )
    )
    resolved_revision = snapshot.name
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
        raise ValueError(f"Expected immutable Hugging Face snapshot revision, received {resolved_revision}")
    seed_everything(seed)
    model = OpenTSLMFlamingo(
        "cuda",
        llm_id=str(snapshot),
        freeze_lm_embeddings=True,
        cross_attn_every_n_layers=1,
        gradient_checkpointing=False,
    ).cuda()
    # BF16 frozen base weights; FP32 master weights for the trainable modules.
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.model.lang_encoder.config.use_cache = False
    model.model.lang_encoder.generation_config.disable_compile = True
    # OpenFlamingo hides this named Gemma argument behind **kwargs. Expose it so
    # HF generation detects the native last-token-only vocabulary projection.
    original_forward = model.model.lang_encoder.forward

    def forward_with_logits(self, input_ids, attention_mask=None, logits_to_keep=0, **kwargs):
        return original_forward(
            input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep, **kwargs
        )

    model.model.lang_encoder.forward = MethodType(forward_with_logits, model.model.lang_encoder)
    trainable = {name: value for name, value in model.model.named_parameters() if value.requires_grad}
    if not trainable or any("decoder_layer." in name for name in trainable):
        raise ValueError("Unexpected trainable frozen-language-model parameters")
    if any(value.requires_grad for value in model.model.lang_encoder.get_input_embeddings().parameters()):
        raise ValueError("The base language model embeddings must remain frozen")
    return model, resolved_revision


def parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, model, config: dict, step: int, score: float) -> None:
    parameters = {
        name: value.detach().cpu() for name, value in model.model.named_parameters() if value.requires_grad
    }
    # Token additions are frozen but initialized when loading the base. Saving
    # their tiny embedding rows makes reconstruction independent of that detail.
    tokenizer = model.text_tokenizer
    special_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("<image>", "<|endofchunk|>")]
    embeddings = model.model.lang_encoder.get_input_embeddings().weight
    payload = {
        "trainable": parameters,
        "config": config,
        "step": step,
        "selection_accuracy": score,
        "special_token_ids": special_ids,
        "special_embeddings": embeddings[special_ids].detach().cpu(),
    }
    partial = path.with_suffix(".partial")
    torch.save(payload, partial)
    partial.replace(path)


def restore_checkpoint(path: Path, model, expected_manifest: str, payload: dict | None = None) -> dict:
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["config"]["manifest_sha256"] != expected_manifest:
        raise ValueError("Checkpoint belongs to a different prepared dataset")
    names = {name for name, value in model.model.named_parameters() if value.requires_grad}
    if names != set(payload["trainable"]):
        raise ValueError("Trainable parameter names differ from checkpoint")
    with torch.no_grad():
        for name, value in model.model.named_parameters():
            if name in payload["trainable"]:
                value.copy_(payload["trainable"][name].to(value.device, value.dtype))
        embeddings = model.model.lang_encoder.get_input_embeddings().weight
        embeddings[payload["special_token_ids"]] = payload["special_embeddings"].to(
            embeddings.device, embeddings.dtype
        )
    return payload


@torch.inference_mode()
def generate_predictions(
    model,
    collator: Collator,
    rows: list[dict],
    *,
    batch_size: int,
    max_new_tokens: int,
    zero: bool = False,
    output_path: Path | None = None,
) -> list[dict]:
    from syntide.baseline.metrics import parse_answer

    model.eval()
    results = []
    started = time.monotonic()
    handle = output_path.with_suffix(".partial").open("w") if output_path else None
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = collator(chunk, training=False, zero=zero)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = model.model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=model.text_tokenizer.eos_token_id,
                    pad_token_id=model.text_tokenizer.pad_token_id,
                    use_cache=True,
                )
            outputs = model.text_tokenizer.batch_decode(
                ids[:, batch["lang_x"].shape[1] :], skip_special_tokens=True
            )
            for row, output in zip(chunk, outputs, strict=True):
                prediction = {
                    key: row[key]
                    for key in (
                        "task_id",
                        "record_id",
                        "subject_uid",
                        "template_id",
                        "is_artifact",
                        "question",
                        "answer",
                        "answer_options",
                    )
                }
                prediction.update(
                    output_text=output, parsed_answer=parse_answer(output, row["answer_options"])
                )
                prediction["invalid"] = prediction["parsed_answer"] is None
                results.append(prediction)
                if handle:
                    handle.write(json.dumps(prediction) + "\n")
            if handle:
                handle.flush()
            if start == 0 or (start // batch_size + 1) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "event": "generation",
                            "completed": len(results),
                            "total": len(rows),
                            "elapsed_s": round(time.monotonic() - started, 2),
                        }
                    ),
                    flush=True,
                )
    finally:
        if handle:
            handle.close()
    if output_path:
        if [row["task_id"] for row in results] != [row["task_id"] for row in rows]:
            raise ValueError("Generated predictions do not cover the requested questions in order")
        output_path.with_suffix(".partial").replace(output_path)
    return results


@torch.inference_mode()
def option_predictions(model, collator: Collator, rows: list[dict], batch_size: int) -> list[dict]:
    """Length-normalized conditional log likelihood over each allowed answer."""
    from syntide.baseline.metrics import parse_answer

    model.eval()
    jobs = [
        (row_index, option_index, option)
        for row_index, row in enumerate(rows)
        for option_index, option in enumerate(row["answer_options"])
    ]
    # Nearby prompt lengths share projected positions and minimize padding.
    jobs.sort(key=lambda job: len(collator.tokenizer.encode(collator.prompt(rows[job[0]])[0])))
    all_scores = [[0.0] * len(row["answer_options"]) for row in rows]
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        batch = collator(
            [rows[item[0]] for item in chunk], training=True, options=[item[2] for item in chunk]
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses, mask = answer_losses(model, batch)
            scores = (-(losses * mask).sum(1) / mask.sum(1)).tolist()
        for (row_index, option_index, _), value in zip(chunk, scores, strict=True):
            all_scores[row_index][option_index] = value
    results = []
    for row, scores in zip(rows, all_scores, strict=True):
        options = row["answer_options"]
        answer = options[int(np.argmax(scores))]
        output_text = f"Answer: {answer}"
        results.append(
            {
                **row,
                "output_text": output_text,
                "parsed_answer": parse_answer(output_text, options),
                "option_log_likelihoods": scores,
            }
        )
    return results


@torch.inference_mode()
def signal_probe(model, collator: Collator, row: dict) -> dict:
    """Hold the entire text prompt fixed; only replace the signal tensor."""
    model.eval()
    batch = collator([row], training=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        real = model.model(**batch).logits[:, -1].float()
        batch["vision_x"] = torch.zeros_like(batch["vision_x"])
        zero = model.model(**batch).logits[:, -1].float()
    difference = (real - zero).abs()
    return {
        "max_abs_logit_difference": difference.max().item(),
        "mean_abs_logit_difference": difference.mean().item(),
        "real_argmax": real.argmax(-1).item(),
        "zero_argmax": zero.argmax(-1).item(),
    }


def train(args) -> None:
    from syntide.baseline import metrics as metrics_module
    from syntide.baseline.metrics import score_predictions

    exposure.validate_train_arguments(args)
    prepared = Prepared(args.prepared, ("train", "selection"))
    baseline = json.loads(args.baseline_config.read_text())
    source_manifest = json.loads(args.source_manifest.read_text())
    source_binding = prepared.manifest.get("exposure_source_manifest", {})
    if (source_binding.get("file") != "source-manifest.json"
            or args.source_manifest.resolve() != (args.prepared / "source-manifest.json").resolve()
            or source_binding.get("sha256") != sha256(args.source_manifest)
            or source_binding.get("bytes") != args.source_manifest.stat().st_size):
        raise ValueError("Prepared dataset does not bind this exact source manifest")
    sources = exposure.validate_sources(prepared.rows["train"], source_manifest)
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        raise ValueError(f"Run already exists: {run}; use a fresh run directory to avoid mixing trials")
    model, revision = load_model(args.seed, args.base_revision)
    collator = Collator(model.text_tokenizer, prepared, args.max_length)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    config = {
        "model": BASE_MODEL,
        "base_revision": revision,
        "opentslm_commit": UPSTREAM_COMMIT,
        "trainer_source_sha256": sha256(Path(__file__)),
        "metrics_source_sha256": sha256(Path(metrics_module.__file__)),
        "manifest_sha256": prepared.manifest_sha256,
        "data_files": prepared.file_sha256,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation + exposure.GENERATED_BATCH,
        "original_effective_batch_size": args.batch_size * args.gradient_accumulation,
        "generated_batch_size": exposure.GENERATED_BATCH,
        "learning_rate_schedule_steps": exposure.LR_SCHEDULE_STEPS,
        "warmup_steps": exposure.WARMUP_STEPS,
        "checkpoint_policy": exposure.CHECKPOINT_POLICY,
        "primary_original_draws": exposure.PRIMARY_STEP * 16,
        "primary_generated_draws": exposure.PRIMARY_STEP * exposure.GENERATED_BATCH,
        "baseline_config_sha256": sha256(args.baseline_config),
        "baseline_trainer_source_sha256": exposure.BASELINE_TRAINER_SHA256,
        "exposure_helper_sha256": sha256(Path(exposure.__file__)),
        "source_manifest_sha256": sha256(args.source_manifest),
        "source_manifest": source_manifest,
        "sampling": "independent NumPy default_rng(0) permutation-and-wrap stream per source",
        "generated_forward_rng": "fork CPU/current CUDA RNG; manual_seed(1000000+step); restore original states",
        "loss_weighting": "existing token-normalized original microbatches weighted 8/18 each; generated microbatch 2/18",
        "selection_policy": "diagnostic only; never chooses an earlier primary checkpoint",
        "val_every": args.val_every,
        "selection_method": args.selection_method,
        "selection_limit": args.selection_limit,
        "selection_questions": len(prepared.rows["selection"]),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "gradient_clip": 1.0,
        "precision": "bf16 autocast; frozen base bf16; trainable weights fp32",
        "prompt_version": PROMPT_VERSION,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "init_sha256": parameter_fingerprint(model),
        "python": os.sys.version,
        "gpu": torch.cuda.get_device_name(),
        "token_lengths": {
            split: token_length_summary(collator, prepared.rows[split]) for split in ("train", "selection")
        },
        "packages": {
            package: package_version(package)
            for package in ("torch", "transformers", "open-flamingo", "opentslm", "numpy", "pyarrow")
        },
    }
    # Model construction and every frozen package/source setting must match the
    # original baseline before any optimization or scientific output is written.
    exposure.validate_baseline(config, baseline)
    write_json(run / "config.json", config)
    print(json.dumps({"event": "configuration", **config}), flush=True)
    if args.probe:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.eval()
            batch = collator(prepared.rows["train"][:1], training=True)
            ordinary_loss = model.model(**batch).loss.item()
            losses, mask = answer_losses(model, batch)
            answer_only_loss = (losses.sum() / mask.sum()).item()
        loss_check = {
            "ordinary_loss": ordinary_loss,
            "answer_only_loss": answer_only_loss,
            "absolute_difference": abs(ordinary_loss - answer_only_loss),
        }
        write_json(run / "loss_equivalence.json", loss_check)
        if not math.isclose(ordinary_loss, answer_only_loss, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"Answer-only loss does not match ordinary loss: {loss_check}")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    original_stream = exposure.PermutationStream(sources["original"], args.seed)
    generated_stream = exposure.PermutationStream(sources["generated"], args.seed)
    history = []
    best_score, best_step = -1.0, 0
    started = time.monotonic()
    training_seconds = 0.0
    selection_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr_scale = exposure.lr_scale(step)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_scale
        step_started = time.monotonic()
        total_loss = 0.0
        for _ in range(args.gradient_accumulation):
            batch_rows = original_stream.take(args.batch_size)
            batch = collator(batch_rows, training=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses, mask = answer_losses(model, batch)
                loss = exposure.weighted_microbatch_loss(losses, mask, args.batch_size)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            total_loss += loss.detach().item()
        # Isolate synthetic forward randomness so later original batches retain
        # the archived baseline RNG chronology, including dropout draws.
        generated_rows = generated_stream.take(exposure.GENERATED_BATCH)
        with exposure.generated_forward_rng(torch, step, [torch.cuda.current_device()]):
            batch = collator(generated_rows, training=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses, mask = answer_losses(model, batch)
                loss = exposure.weighted_microbatch_loss(losses, mask, exposure.GENERATED_BATCH)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite generated loss at step {step}")
            loss.backward()
            total_loss += loss.detach().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds = time.monotonic() - step_started
        training_seconds += step_seconds
        if step == 1 or step % args.log_every == 0:
            progress = {
                "event": "train",
                "step": step,
                "loss": total_loss,
                "lr": group["lr"],
                "grad_norm": float(grad_norm),
                "step_s": step_seconds,
                "elapsed_s": time.monotonic() - started,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                "source_exposure": {
                    "original": original_stream.audit(), "generated": generated_stream.audit()
                },
            }
            print(json.dumps(progress), flush=True)
            with (run / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(progress) + "\n")
        if step % args.val_every == 0 or step == args.max_steps:
            val_started = time.monotonic()
            rows = prepared.rows["selection"]
            if args.selection_limit:
                rows = rows[: args.selection_limit]
            if args.selection_method == "generation":
                predictions = generate_predictions(
                    model, collator, rows, batch_size=args.eval_batch_size, max_new_tokens=args.max_new_tokens
                )
            else:
                predictions = option_predictions(model, collator, rows, args.eval_batch_size)
            scores = score_predictions(predictions)
            score = scores["groups"]["all"]["accuracy"]
            elapsed = time.monotonic() - val_started
            selection_seconds += elapsed
            event = {
                "event": "selection",
                "step": step,
                "accuracy": score,
                "elapsed_s": elapsed,
                "questions": len(rows),
                "scores": scores,
            }
            history.append(event)
            print(json.dumps({k: v for k, v in event.items() if k != "scores"}), flush=True)
            write_json(run / "selection_history.json", history)
            if score > best_score:
                best_score, best_step = score, step
                write_json(run / "diagnostic_peak_selection_metrics.json", scores)
            if step == exposure.PRIMARY_STEP:
                write_json(run / "primary_selection_metrics.json", scores)
        if args.probe and (step == 2 or step == args.max_steps):
            probe = signal_probe(model, collator, prepared.rows["selection"][0])
            write_json(run / f"signal_probe_step{step}.json", probe)
            print(json.dumps({"event": "signal_probe", "step": step, **probe}), flush=True)
            if step == args.max_steps and step >= 10 and probe["max_abs_logit_difference"] == 0:
                raise ValueError(
                    "Signal probe failed: real and zero ECG have identical logits after training"
                )
    source_exposure = {"original": original_stream.audit(), "generated": generated_stream.audit()}
    if (source_exposure["original"]["draws"] != exposure.PRIMARY_STEP * 16
            or source_exposure["generated"]["draws"] != exposure.PRIMARY_STEP * exposure.GENERATED_BATCH):
        raise ValueError("Actual source draws differ from the fixed exposure budget")
    save_checkpoint(run / "primary.pt", model, config, args.max_steps, history[-1]["accuracy"])
    result = {
        "status": "trained",
        "seed": args.seed,
        "primary_step": args.max_steps,
        "primary_selection_accuracy": history[-1]["accuracy"],
        "diagnostic_peak_step": best_step,
        "diagnostic_peak_selection_accuracy": best_score,
        "checkpoint_policy": exposure.CHECKPOINT_POLICY,
        "source_exposure": source_exposure,
        "max_steps": args.max_steps,
        "training_seconds": training_seconds,
        "selection_seconds": selection_seconds,
        "wall_seconds": time.monotonic() - started,
        "mean_step_seconds": training_seconds / args.max_steps,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "primary_checkpoint_sha256": sha256(run / "primary.pt"),
        "smoke_only": bool(args.selection_limit),
        "manifest_sha256": prepared.manifest_sha256,
        "distinct_seen_training_questions": len(collator.training_token_cache),
        "distinct_truncated_rationales": len(collator.truncated_tasks),
    }
    write_json(run / "train_result.json", result)
    print(json.dumps(result), flush=True)


def read_saved_predictions(path: Path, expected_rows: list[dict]) -> list[dict]:
    """Recover only a complete, ordered, unmodified evaluation result."""
    from syntide.baseline.metrics import parse_answer

    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    if len(predictions) != len(expected_rows):
        raise ValueError("Saved predictions do not cover the complete requested split")
    fields = (
        "task_id",
        "record_id",
        "subject_uid",
        "template_id",
        "is_artifact",
        "question",
        "answer",
        "answer_options",
    )
    for prediction, expected in zip(predictions, expected_rows, strict=True):
        if any(prediction.get(field) != expected[field] for field in fields):
            raise ValueError(f"Saved prediction metadata/order differs for {expected['task_id']}")
        if not isinstance(prediction.get("output_text"), str):
            raise ValueError(f"Saved output text missing for {expected['task_id']}")
        parsed = parse_answer(prediction["output_text"], expected["answer_options"])
        if prediction.get("parsed_answer") != parsed or prediction.get("invalid") is not (parsed is None):
            raise ValueError(f"Saved parsed answer disagrees with output for {expected['task_id']}")
    return predictions


def evaluate(args) -> None:
    from syntide.baseline import metrics as metrics_module
    from syntide.baseline.metrics import score_predictions

    prepared = Prepared(args.prepared, (args.split,))
    config = json.loads((args.run_dir / "config.json").read_text())
    completed = json.loads((args.run_dir / "train_result.json").read_text())
    if completed["status"] != "trained" or completed["max_steps"] != config["max_steps"]:
        raise ValueError("Evaluation requires a completed fixed-budget training run")
    if config["trainer_source_sha256"] != sha256(Path(__file__)):
        raise ValueError("Trainer source changed since this checkpoint was trained")
    if config.get("exposure_helper_sha256") != sha256(Path(exposure.__file__)):
        raise ValueError("Exposure helper source changed since this checkpoint was trained")
    if config["metrics_source_sha256"] != sha256(Path(metrics_module.__file__)):
        raise ValueError("Metrics source changed since this checkpoint was trained")
    if config["manifest_sha256"] != prepared.manifest_sha256:
        raise ValueError("Evaluation dataset manifest differs from training")
    if config["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Checkpoint prompt version differs from this evaluator")
    if config["selection_limit"] and not args.limit:
        raise ValueError("A smoke checkpoint can only be used for --limit smoke evaluation")
    checkpoint = args.run_dir / "primary.pt"
    checkpoint_sha256 = sha256(checkpoint)
    if checkpoint_sha256 != completed["primary_checkpoint_sha256"]:
        raise ValueError("Primary checkpoint checksum differs from the completed training run")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["config"] != config:
        raise ValueError("Checkpoint configuration differs from saved run configuration")
    history = json.loads((args.run_dir / "selection_history.json").read_text())
    exposure.validate_primary_checkpoint(config, completed, payload, history)
    rows = prepared.rows[args.split]
    if args.limit:
        rows = rows[: args.limit]
    suffix = f"{args.split}{'_zero' if args.zero_signal else ''}{'_smoke' if args.limit else ''}"
    output = args.run_dir / f"{suffix}_predictions.jsonl"
    metrics_path = args.run_dir / f"{suffix}_metrics.json"
    claim_path = args.run_dir / f"{suffix}_claim.json"
    identity = {
        "split": args.split,
        "zero_signal": args.zero_signal,
        "seed": config["seed"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": payload["step"],
        "manifest_sha256": prepared.manifest_sha256,
        "config_sha256": sha256(args.run_dir / "config.json"),
        "trainer_source_sha256": config["trainer_source_sha256"],
        "exposure_helper_sha256": config["exposure_helper_sha256"],
        "checkpoint_policy": config["checkpoint_policy"],
        "metrics_source_sha256": config["metrics_source_sha256"],
        "questions": len(rows),
        "task_ids_sha256": hashlib.sha256(json.dumps([row["task_id"] for row in rows]).encode()).hexdigest(),
        "max_length": config["max_length"],
        "max_new_tokens": config["max_new_tokens"],
    }
    if claim_path.exists():
        claim = json.loads(claim_path.read_text())
        if claim["identity"] != identity:
            raise ValueError("Existing evaluation claim has different data/model/generation settings")
    else:
        if output.exists() or metrics_path.exists():
            raise ValueError("Existing evaluation output has no provenance claim; refusing recovery")
        claim = {"identity": identity, "started_at_epoch_s": time.time(), "batch_size": args.batch_size}
        write_json(claim_path, claim)

    recovered = output.exists()
    if recovered:
        # A crash after atomic JSONL publication must never trigger another pass
        # over the test fold. Validate and score the exact saved outputs only.
        predictions = read_saved_predictions(output, rows)
    else:
        if metrics_path.exists():
            raise ValueError("Evaluation metrics exist but their prediction artifact is missing")
        model, revision = load_model(config["seed"], config["base_revision"])
        if revision != config["base_revision"]:
            raise ValueError("Base model revision differs from training")
        restore_checkpoint(checkpoint, model, prepared.manifest_sha256, payload=payload)
        collator = Collator(model.text_tokenizer, prepared, config["max_length"])
        predictions = generate_predictions(
            model,
            collator,
            rows,
            batch_size=claim["batch_size"],
            max_new_tokens=config["max_new_tokens"],
            zero=args.zero_signal,
            output_path=output,
        )
    metrics = {
        "split": args.split,
        "zero_signal": args.zero_signal,
        "seed": config["seed"],
        "checkpoint_step": payload["step"],
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": prepared.manifest_sha256,
        "prediction_sha256": sha256(output),
        "claim_sha256": sha256(claim_path),
        "elapsed_s": time.time() - claim["started_at_epoch_s"],
        "metrics_recovered_from_predictions": recovered,
        "questions": len(rows),
        "strict": score_predictions(predictions),
        "upstream_parser": score_predictions(predictions, parser="upstream"),
    }
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        stable_keys = set(metrics) - {"elapsed_s", "metrics_recovered_from_predictions"}
        if any(saved.get(key) != metrics[key] for key in stable_keys):
            raise ValueError("Existing metrics disagree with verified predictions/provenance")
        metrics = saved
    else:
        write_json(metrics_path, metrics)
    print(
        json.dumps({k: v for k, v in metrics.items() if k not in ("strict", "upstream_parser")}), flush=True
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    score = commands.add_parser("evaluate")
    for command in (fit, score):
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--baseline-config", type=Path, required=True)
    fit.add_argument("--source-manifest", type=Path, required=True)
    fit.add_argument("--max-steps", type=int, required=True)
    fit.add_argument("--val-every", type=int, default=500)
    fit.add_argument("--gradient-accumulation", type=int, default=1)
    fit.add_argument("--eval-batch-size", type=int, default=16)
    fit.add_argument("--learning-rate", type=float, default=2e-4)
    fit.add_argument("--max-length", type=int, default=1536)
    fit.add_argument("--max-new-tokens", type=int, default=384)
    fit.add_argument("--base-revision", default="main")
    fit.add_argument("--selection-method", choices=("generation", "option-nll"), default="generation")
    fit.add_argument("--selection-limit", type=int, default=0, help="Smoke only: limit selection questions")
    fit.add_argument(
        "--probe", action="store_true", help="On CUDA, compare real and zero signal logits after updates"
    )
    fit.add_argument("--log-every", type=int, default=20)
    score.add_argument("--split", choices=("selection", "validation", "test"), required=True)
    score.add_argument("--zero-signal", action="store_true")
    score.add_argument("--limit", type=int, default=0, help="Smoke only: limit evaluation questions")
    args = parser.parse_args(argv)
    for name in (
        "batch_size",
        "max_steps",
        "val_every",
        "gradient_accumulation",
        "eval_batch_size",
        "max_length",
        "max_new_tokens",
        "log_every",
    ):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    require_cuda()
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
