"""Format adapter: ``syntide.qa_trial.QAExample`` -> the RunPod baseline's OpenTSLM-Flamingo prompt.

The baseline loop (branch ``runpod-baseline``, ``opentslm_loop.make_prompt`` / ``target_text``) used
prompt version ``ecgqa-cot-single-joint-media-v1``: upstream ECG wording, per-lead z-scores after the
signal is in mV, the per-lead mean and std written into the text, one joint ``<image>`` marker for the
12 leads, the clinical context, the question and the comma-separated options, and the target
``"<rationale>\nAnswer: <answer>"`` with only the target tokens in the loss.

``prompt`` reproduces that text exactly for a single-recording question with options (every original
ECG-QA-CoT task), so arm 1 is the baseline's input. Two documented extensions cover new task shapes:

- ``-numeric`` (no options): the options line asks for a number in the question's units;
- ``-pair`` (two recordings): one joint marker per recording, labelled "Recording A" / "Recording B".

The target is ``QAExample.target`` unchanged; ``loss_spans`` says which characters are supervised.
Nothing here imports OpenTSLM or torch; this is checked against the baseline text only by unit tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from syntide.data.schema import LEADS
from syntide.qa_trial import QAExample

PROMPT_VERSION = "ecgqa-cot-single-joint-media-v1"
_PARAGRAPH = (
    "- Write your rationale as a single, natural paragraph — "
    "do not use bullet points, numbered steps, or section headings."
)


def _leads(signal: np.ndarray, channels: list[str]) -> tuple[str, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError("nonfinite ECG")
    if list(channels) != list(LEADS):
        raise ValueError(f"the baseline prompt expects leads {LEADS}, got {channels}")
    means = signal.mean(axis=-1, keepdims=True)
    stds = signal.std(axis=-1, keepdims=True)
    normalized = (signal - means) / np.where(stds > 1e-6, stds, 1.0)
    text = "\n".join(
        f"This is ECG Lead {name}, it has mean {float(means[i, 0]):.4f} and std {float(stds[i, 0]):.4f}:"
        for i, name in enumerate(LEADS)
    )
    return text, np.ascontiguousarray(normalized)


def prompt(example: QAExample) -> dict[str, Any]:
    """The prompt text, the normalized signals (one array per ``<image>`` marker) and the version."""
    student = example.student
    contexts = student["context"].get("clinical_context", [""])
    blocks, signals = [], []
    for k, (signal, ref) in enumerate(zip(student["signals"], student["inputs"], strict=True)):
        text, normalized = _leads(signal, ref["signal"]["channels"])
        label = "" if len(student["signals"]) == 1 else f"Recording {'AB'[k]}: "
        blocks.append(f"<image> {label}{text} <|endofchunk|>")
        signals.append(normalized)
    version = PROMPT_VERSION
    if len(signals) > 1:
        version += "-pair"
    if student["options"]:
        choose = f"select your answer from the following options:\n{', '.join(student['options'])}"
    else:
        version += "-numeric"
        choose = "give your answer as a number in the units the question asks for."
    context = (
        contexts[0]
        if len(contexts) == 1
        else " | ".join(f"Recording {'AB'[k]}: {c}" for k, c in enumerate(contexts))
    )
    text = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context: {context}

Your task is to examine the ECG signal and answer the following medical question:

Question: {student["question"]}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
{_PARAGRAPH}
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question.

{chr(10).join(blocks)}

Based on your analysis of the ECG data, {choose}

- Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""
    return {"text": text, "signals": signals, "prompt_version": version}


def loss_spans(example: QAExample) -> dict[str, Any]:
    """The supervised text (after a single leading space, as in the baseline) and its character span."""
    target = " " + example.target
    return {"target": target, "supervised": [0, len(target)], "mode": example.target_mode}
