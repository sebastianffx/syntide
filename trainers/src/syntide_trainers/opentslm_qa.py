"""Format trial examples as the original OpenTSLM-Flamingo training prompt.

Preserve the baseline prompt text for single-recording ECG-QA-CoT questions.
Convert signals to millivolts, subtract each lead's mean, and divide by its
standard deviation. Include those original statistics, one image marker for
all twelve leads, clinical context, the question, and answer options.

Numeric questions use the question's units instead of answer options. Paired
questions receive one image marker per recording, labelled Recording A and
Recording B. Training supervises only the supplied target text, preserving
its explanation and final answer line. This module imports neither OpenTSLM
nor PyTorch.
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
        raise ValueError(
            f"the baseline prompt expects leads {LEADS}, got {channels}"
        )
    means = signal.mean(axis=-1, keepdims=True)
    stds = signal.std(axis=-1, keepdims=True)
    normalized = (signal - means) / np.where(stds > 1e-6, stds, 1.0)
    text = "\n".join(
        (
            f"This is ECG Lead {name}, it has mean {float(means[i, 0]):.4f}"
            f" and std {float(stds[i, 0]):.4f}:"
        )
        for i, name in enumerate(LEADS)
    )
    return text, np.ascontiguousarray(normalized)


def prompt(example: QAExample) -> dict[str, Any]:
    """Return prompt text, normalized signals, and the format version."""
    student = example.student
    contexts = student["context"].get("clinical_context", [""])
    blocks, signals = [], []
    for k, (signal, ref) in enumerate(
        zip(student["signals"], student["inputs"], strict=True)
    ):
        text, normalized = _leads(signal, ref["signal"]["channels"])
        label = "" if len(student["signals"]) == 1 else f"Recording {'AB'[k]}: "
        blocks.append(f"<image> {label}{text} <|endofchunk|>")
        signals.append(normalized)
    version = PROMPT_VERSION
    if len(signals) > 1:
        version += "-pair"
    if student["options"]:
        choose = (
            "select your answer from the following options:\n"
            f"{', '.join(student['options'])}"
        )
    else:
        version += "-numeric"
        choose = (
            "give your answer as a number in the units the question asks for."
        )
    context = (
        contexts[0]
        if len(contexts) == 1
        else " | ".join(
            f"Recording {'AB'[k]}: {c}" for k, c in enumerate(contexts)
        )
    )
    text = (
        "You are an expert cardiologist analyzing an ECG "
        f"(electrocardiogram).\n\nClinical Context: {context}\n\nYour task "
        "is to examine the ECG signal and answer the following medical "
        f"question:\n\nQuestion: {student['question']}\n\nInstructions:\n- "
        "Begin by analyzing the time series without assuming a specific "
        "answer.\n- Think step-by-step about what the observed patterns "
        f"suggest regarding the cardiac condition.\n{_PARAGRAPH}\n- Do "
        "**not** mention any final answer until the very end.\n- Consider "
        "the ECG morphology, intervals, and any abnormalities that relate "
        f"to the question.\n\n{chr(10).join(blocks)}\n\nBased on your "
        f"analysis of the ECG data, {choose}\n\n- Make sure that your last "
        'word is the answer. You MUST end your response with "Answer: "\n'
    )
    return {"text": text, "signals": signals, "prompt_version": version}


def loss_spans(example: QAExample) -> dict[str, Any]:
    """Return supervised text and its character range."""
    target = " " + example.target
    return {
        "target": target,
        "supervised": [0, len(target)],
        "mode": example.target_mode,
    }
