"""Synthetic test fixtures for the QA pipeline.

The ECG-like signals here are Gaussian bumps at known beat times plus noise: test fixtures with known
heart rates, NOT clinically validated ECGs and not synthetic data for training.
"""

from __future__ import annotations

import numpy as np

from syntide.data.qa import SignalInfo
from syntide.data.schema import LEADS
from syntide.generate.qa.sources import QASource, SourceRow

FIXTURE_NOTICE = "test fixture with known beat times; not a clinically validated ECG"


def fixture_ecg(
    hr: float,
    *,
    seconds: float = 10.0,
    fs: float = 100.0,
    seed: int = 0,
    jitter: float = 0.0,
    noise: float = 0.02,
    wander: float = 0.0,
    flat: tuple[str, ...] = (),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * fs)) / fs
    beats, b = [], 0.3
    while b < seconds - 0.1:
        beats.append(b)
        b += (60 / hr) * (1 + jitter * rng.standard_normal())
    gains = np.linspace(0.4, 1.2, 12) * np.where(np.arange(12) % 5 == 3, -1, 1)
    sig = np.zeros((12, t.size))
    for beat in beats:
        sig += (
            gains[:, None]
            * (
                np.exp(-0.5 * ((t - beat) / 0.012) ** 2)
                + 0.25 * np.exp(-0.5 * ((t - beat - 0.28) / 0.05) ** 2)
            )[None]
        )
    sig += (
        noise * rng.standard_normal(sig.shape)
        + wander * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6))[None]
    )
    for lead in flat:
        sig[LEADS.index(lead)] = 0.0
    return sig.astype(np.float32)


def question(
    record_id: str, k: int, template_id: int, text: str, answer: str, options: list[str], context: str
) -> dict:
    return {
        "task_id": f"ecgqa-train-{record_id}-{k}",
        "record_id": record_id,
        "question": text,
        "answer": answer,
        "answer_options": options,
        "clinical_context": context,
        "rationale": f"The waveform shows it.\nAnswer: {answer}",
        "template_id": template_id,
        "question_type": "single-verify",
        "cot_split": "train",
        "is_artifact": False,
    }


def memory_source(n: int = 8, *, start: int = 1, purpose: str = "training", seed: int = 0) -> QASource:
    """A small in-memory population: patient i owns record ptbxl-(start+i)."""
    rates = [55, 72, 88, 110, 64, 95, 130, 48, 76, 102]
    rows, signals = [], {}
    for i in range(n):
        record_id = f"ptbxl-{start + i}"
        drift = i % 3 == 0
        quality = "baseline drift noted" if drift else "no issues"
        context = f"{50 + i}-year-old patient. 12-lead ECG. Signal quality: {quality}."
        rows.append(
            SourceRow(
                record_id=record_id,
                subject_uid=f"ptb-xl:{1000 + start + i}",
                labels=["NORM"] if i % 2 else ["MI", "STTC"],
                artifact_types=["baseline_drift"] if drift else [],
                clinical_context=context,
                questions=[
                    question(
                        record_id,
                        0,
                        27,
                        "Does this ECG show baseline drift?",
                        "yes" if drift else "no",
                        ["no", "yes"],
                        context,
                    ),
                    question(
                        record_id,
                        1,
                        4,
                        "Which diagnostic symptom does this ECG show?",
                        "none",
                        ["none", "myocardial infarction"],
                        context,
                    ),
                ],
            )
        )
        signals[record_id] = fixture_ecg(
            rates[i % len(rates)],
            seed=seed + start + i,
            wander=0.8 if drift else 0.05,
            jitter=0.25 if i == 6 else 0.0,
        )

    def load(ids: list[str]) -> np.ndarray:
        return np.stack([signals[r] for r in ids])

    provenance = {
        "fingerprint": f"memory-{start}-{n}-{seed}",
        "role": "pool" if purpose == "training" else "val",
        "purpose": purpose,
        "notice": FIXTURE_NOTICE,
    }
    return QASource(rows, load, provenance, purpose=purpose, signal=SignalInfo(LEADS, 100.0, "mV", 1000))
