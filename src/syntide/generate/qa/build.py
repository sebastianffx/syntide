"""Build and publish a question-answer corpus from verified signal evidence.

The builder processes source recordings and generated signals in bounded units.
Each completed unit records its tasks, evidence, signal references, and
rejected
attempts. An interrupted build resumes from those records. Publication occurs
only after every expected unit passes the corpus checks.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from syntide import __version__
from syntide.data.qa import Evidence, InputRef, QATask, SignalInfo
from syntide.generate import editor
from syntide.generate.qa import acquisition, generated, pairs
from syntide.generate.qa.corpus import CorpusWriter, ledger_row, rejection
from syntide.generate.qa.evidence import (
    EvidenceCache,
    EvidenceExtractor,
    ExtractorConfig,
)
from syntide.generate.qa.evol import GENERATOR as TSEVOL
from syntide.generate.qa.evol import (
    EvolConfig,
    evolve_waveform,
    paraphrase,
    stream,
)
from syntide.generate.qa.sources import QASource, source_tasks
from syntide.generate.qa.teacher import make_teacher
from syntide.generate.qa.templates import TEMPLATES, View
from syntide.generate.qa.verify import CONTEXT_POLICY, student_context

log = logging.getLogger(__name__)
COMPONENTS = ("source", "paraphrase", "grounded", "acquisition", "generated")
SUITE_ROLES = ("selection", "development", "test", "external")


@dataclass
class PairsConfig:
    natural_per_unit: int = 4
    variations: list[str] = field(
        default_factory=lambda: ["amplitude_up@1", "wander_up@1"]
    )
    variation_fraction: float = (
        0.25  # share of recordings that get one controlled variation pair
    )


@dataclass
class GeneratedConfig:
    bundles: list[str] = field(default_factory=list)
    allow_resample: bool = False
    required: list[str] = field(
        default_factory=lambda: [
            "heart_rate_bpm",
            "beat_count",
            "rr_regularity",
        ]
    )
    templates: list[str] = field(
        default_factory=lambda: [
            "rhythm.heart_rate@1",
            "rhythm.beat_count@1",
            "rhythm.regularity@1",
            "rhythm.rate_band@1",
            "acquisition.duration@1",
            "acquisition.flat_leads@1",
        ]
    )
    min_confidence: float = 0.8
    heart_rate_pairs: bool = True


@dataclass
class BuildConfig:
    name: str = "qa"
    purpose: str = "training"
    seed: int = 0
    components: list[str] = field(
        default_factory=lambda: ["source", "grounded"]
    )
    unit_size: int = 64
    max_records: int | None = None
    # evaluation suites keep one task per (record, question), lowest task id
    source_dedupe: bool = False
    paraphrase_fraction: float = (
        1.0  # share of source tasks that get one paraphrase attempt
    )
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    evol: EvolConfig = field(default_factory=EvolConfig)
    acquisition: acquisition.AcquisitionConfig = field(
        default_factory=acquisition.AcquisitionConfig
    )
    pairs: PairsConfig = field(default_factory=PairsConfig)
    generated: GeneratedConfig = field(default_factory=GeneratedConfig)
    teacher: dict[str, Any] | None = None
    cache_dir: str | None = None
    # Evaluation suites: selection | development | test | external. A
    # development suite's predictions may
    # drive a new generation plan; the selection and test suites' never do.
    suite_role: str | None = None
    # Restrict source tasks to an exact list (e.g. the RunPod baseline's
    # selection.parquet): a parquet with
    # a task_id column or a JSON list, pinned by its file sha256. Only
    # recordings with listed tasks are read.
    task_ids_file: str | None = None
    task_ids_sha256: str | None = None
    exclude_corpora: list[str] = field(
        default_factory=list
    )  # whole patients of these corpora are left out


class Quotas:
    """Per-template task limits from an optional generation plan."""

    def __init__(self, plan: dict[str, Any] | None) -> None:
        self.limits = dict((plan or {}).get("template_quotas", {}))
        self.used: Counter = Counter()

    def allow(self, task: QATask) -> bool:
        if not self.limits:
            return True
        limit = self.limits.get(task.template, 0)
        if self.used[task.template] >= limit:
            return False
        self.used[task.template] += 1
        return True


def identity(
    cfg: BuildConfig,
    source: QASource,
    plan: dict[str, Any] | None,
    parent: str | None,
) -> dict[str, Any]:
    unknown = sorted(set(cfg.components) - set(COMPONENTS))
    if unknown:
        raise ValueError(
            f"unknown components {unknown}; choose from {COMPONENTS}"
        )
    if cfg.purpose != source.purpose:
        raise PermissionError(
            f"a {cfg.purpose} corpus cannot be built from a "
            f"{source.purpose} source"
        )
    if cfg.purpose == "evaluation" and set(cfg.components) - {
        "source",
        "grounded",
    }:
        raise ValueError(
            "evaluation suites hold source or grounded tasks on "
            "held-out recordings only"
        )
    if (cfg.purpose == "evaluation") != (cfg.suite_role is not None):
        raise ValueError(
            "an evaluation suite needs a suite_role, and only an "
            "evaluation suite has one"
        )
    if cfg.suite_role is not None and cfg.suite_role not in SUITE_ROLES:
        raise ValueError(f"suite_role must be one of {SUITE_ROLES}")
    teacher = cfg.teacher or {}
    return {
        "schema": "syntide.qa-build@1",
        "name": cfg.name,
        "purpose": cfg.purpose,
        "components": sorted(cfg.components),
        "config": {
            **asdict(cfg),
            "cache_dir": None,
            "teacher": {
                k: v for k, v in teacher.items() if k not in ("store", "mode")
            },
        },
        "source": source.provenance,
        "context_policy": CONTEXT_POLICY,
        "templates": sorted(TEMPLATES),
        "generators": {
            "tsevol": TSEVOL,
            "acquisition": f"{acquisition.GENERATOR}@{acquisition.VERSION}",
            "editor": f"{editor.GENERATOR}@{editor.VERSION}",
            "pairs": "syntide.pairs@1",
        },
        "plan_id": (plan or {}).get("plan_id"),
        "parent_corpus_id": parent,
        "syntide_version": __version__,
    }


def build_corpus(
    cfg: BuildConfig,
    source: QASource,
    output: str | Path,
    *,
    plan: dict[str, Any] | None = None,
    parent_corpus_id: str | None = None,
    tracker: Any = None,
    stop_after_units: int | None = None,
) -> dict[str, Any] | None:
    """Build and publish a corpus, resuming completed work when available.

    Args:
        cfg: Corpus construction settings.
        source: Checked source recordings and their loader.
        output: New directory for the completed corpus.
        plan: Optional task allocation plan.
        parent_corpus_id: Identifier of the corpus this build extends.
        tracker: Optional progress and metric logger.
        stop_after_units: Stop after this many units to simulate interruption.

    Returns:
        Published corpus metadata, or ``None`` after a simulated interruption.
    """
    if cfg.exclude_corpora:
        from syntide.generate.qa.corpus import load_corpus

        excluded = {
            subject
            for path in cfg.exclude_corpora
            for ref in load_corpus(path).inputs.values()
            for subject in ref.root_subject_uids
        }
        source = source.without_subjects(excluded)
    if plan is not None:
        from syntide.generate.qa.allocation import verify_plan

        verify_plan(plan)
    keep_ids = _task_ids(cfg)
    if keep_ids is not None:
        listed = {
            r.record_id
            for r in source.rows
            if any(q["task_id"] in keep_ids for q in r.questions)
        }
        found = {
            q["task_id"] for r in source.rows for q in r.questions
        } & keep_ids
        if found != keep_ids:
            raise PermissionError(
                f"{len(keep_ids - found)} listed tasks are not "
                f"questions of this source role"
            )
        source = source.restricted_to(listed)
    ident = identity(cfg, source, plan, parent_corpus_id)
    writer = CorpusWriter(output, ident)
    teacher = make_teacher(cfg.teacher)
    if "paraphrase" in cfg.components and teacher is None:
        raise ValueError(
            "the paraphrase component needs a teacher (provider "
            "template or anthropic)"
        )
    cache = EvidenceCache(cfg.cache_dir) if cfg.cache_dir else None
    extractor = EvidenceExtractor(cfg.extractor, cache)
    quotas = Quotas(plan)
    for chunk in (
        writer.chunks()
    ):  # a resumed build continues the plan's quotas where it stopped
        for task in chunk["tasks"]:
            quotas.used[task["template"]] += 1
    records = [row.record_id for row in source.rows][: cfg.max_records]
    units = [
        (f"real-{i:05d}", records[start : start + cfg.unit_size])
        for i, start in enumerate(range(0, len(records), cfg.unit_size))
    ]
    expected = [name for name, _ in units]
    if "generated" in cfg.components:
        expected += [
            "generated"
        ]  # one unit, so same-seed pairs can span bundles
    done = 0
    totals: Counter = Counter()
    for name, ids in units:
        if writer.unit_done(name):
            continue
        result = _real_unit(
            name, ids, cfg, source, extractor, teacher, quotas, writer, keep_ids
        )
        writer.write_unit(name, *result)
        done += 1
        _log_unit(tracker, totals, name, result, done)
        if stop_after_units is not None and done >= stop_after_units:
            return None
    if "generated" in cfg.components and not writer.unit_done("generated"):
        result = _generated_unit(
            "generated",
            [Path(b) for b in cfg.generated.bundles],
            cfg,
            source,
            extractor,
            quotas,
            writer,
        )
        writer.write_unit("generated", *result)
        done += 1
        _log_unit(tracker, totals, "generated", result, done)
    # Defense in depth: every input of every committed task, against the
    # source's role.
    source.check_ancestry(
        [
            (
                ref["waveform_id"],
                ref["root_record_ids"],
                ref["root_subject_uids"],
            )
            for chunk in writer.chunks()
            for task in chunk["tasks"]
            for ref in task["inputs"]
        ]
    )
    info = writer.publish(expected_units=expected)
    if teacher is not None:
        log.info(
            "teacher store %s (sha %s)",
            teacher.store.path,
            teacher.store.sha256(),
        )
    if tracker is not None:
        tracker.log_summary(
            {"corpus": cfg.name, "purpose": cfg.purpose},
            {
                f"corpus/{k}": v
                for k, v in info["counts"].items()
                if isinstance(v, int | float)
            },
        )
        tracker.close()
    return info


def _log_unit(
    tracker: Any, totals: Counter, name: str, result: tuple, step: int
) -> None:
    tasks, _, ledger, _ = result
    totals["accepted"] += sum(
        1 for t in tasks if t.verification.get("status") == "accepted"
    )
    totals["attempts"] += len(ledger)
    log.info(
        "%s: %d tasks accepted of %d attempts", name, len(tasks), len(ledger)
    )
    if tracker is not None:
        tracker.log_scalars(
            {
                "build/accepted": totals["accepted"],
                "build/attempts": totals["attempts"],
                "build/acceptance_rate": totals["accepted"]
                / max(totals["attempts"], 1),
            },
            step,
        )


def _real_unit(
    name: str,
    ids: list[str],
    cfg: BuildConfig,
    source: QASource,
    extractor: EvidenceExtractor,
    teacher: Any,
    quotas: Quotas,
    writer: CorpusWriter,
    keep_ids: set[str] | None = None,
) -> tuple[list[QATask], list[Evidence], list[dict[str, Any]], list[InputRef]]:
    comps = set(cfg.components)
    tasks: list[QATask] = []
    evidence: list[Evidence] = []
    ledger: list[dict[str, Any]] = []
    waveforms: list[InputRef] = []
    views: dict[str, View] = {}
    contexts: dict[str, str] = {}

    def record(task: QATask, attempt: str) -> None:
        if task.verification.get("status") == "accepted" and not quotas.allow(
            task
        ):
            task = _reject(task, "plan_quota_reached")
        ledger.append(ledger_row(task, name, attempt))
        if task.verification.get("status") == "accepted":
            tasks.append(task)

    for ref, x in source.load(
        ids
    ):  # the source role was checked when it was opened
        row = source.row(ref.waveform_id)
        sha = ref.waveform_sha256
        writer.put_signal(ref, x)
        waveforms.append(ref)
        annotations = {
            "labels": row.labels,
            "artifact_types": row.artifact_types,
        }
        ev = extractor.extract(x, ref, annotations)
        evidence.extend(ev)
        views[ref.waveform_id], contexts[ref.waveform_id] = (
            View(ref, ev),
            row.clinical_context,
        )
        if comps & {"source", "paraphrase"}:
            originals = source_tasks(source, ref, dedupe=cfg.source_dedupe)
            if keep_ids is not None:
                originals = [
                    t
                    for t in originals
                    if t.lineage["source_task_id"] in keep_ids
                ]
            if "source" in comps:
                for task in originals:
                    record(task, "source")
            if "paraphrase" in comps:
                rng = stream(cfg.seed, "paraphrase", sha)
                for task in originals:
                    if (
                        rng.random() < cfg.paraphrase_fraction
                        and (made := paraphrase(task, teacher)) is not None
                    ):
                        record(made, "paraphrase")
        if not comps & {"grounded", "acquisition"}:
            continue
        context = {
            "clinical_context": [student_context("real", row.clinical_context)]
        }
        grown = evolve_waveform(
            views[ref.waveform_id],
            ev,
            context,
            cfg.evol,
            cfg.seed,
            teacher=teacher if "grounded" in comps else None,
        )
        evidence.extend(grown.evidence)
        if "grounded" in comps:
            for task in grown.accepted:
                record(task, "grounded")
            for task in grown.rejected:
                ledger.append(ledger_row(task, name, "grounded"))
        if "acquisition" in comps:
            _acquisition(
                name,
                ref,
                x,
                ev,
                row.clinical_context,
                grown.accepted,
                cfg,
                extractor,
                writer,
                record,
                ledger,
                evidence,
                waveforms,
            )
    if "grounded" in comps and len(views) > 1:
        rng = stream(
            cfg.seed, "pair", views[ids[0]].ref.waveform_sha256, len(ids)
        )
        for a, b in pairs.natural_partners(
            ids, cfg.pairs.natural_per_unit, rng
        ):
            accepted, rejected, cited = pairs.comparison_tasks(
                views[a], views[b], [contexts[a], contexts[b]], rng
            )
            evidence.extend(cited)
            for task in accepted:
                record(task, "natural_pair")
            for task in rejected:
                ledger.append(ledger_row(task, name, "natural_pair"))
    return tasks, evidence, ledger, waveforms


def _acquisition(
    unit: str,
    ref: InputRef,
    x: np.ndarray,
    ev: list[Evidence],
    context: str,
    grounded: list[QATask],
    cfg: BuildConfig,
    extractor: EvidenceExtractor,
    writer: CorpusWriter,
    record: Any,
    ledger: list[dict[str, Any]],
    evidence: list[Evidence],
    waveforms: list[InputRef],
) -> None:
    acq = cfg.acquisition
    rng = stream(cfg.seed, "acquisition", ref.waveform_sha256)
    carried_from = [t for t in grounded if t.paraphrase == "seed"]
    for k in range(acq.per_waveform):
        op = acq.operations[int(rng.integers(len(acq.operations)))]
        try:
            recipe = acquisition.draw_recipe(op, acq, rng, ref.signal)
            variant = acquisition.apply(x, ref, recipe, acq)
        except acquisition.UnsupportedTransformation as error:
            ledger.append(
                rejection(
                    unit,
                    f"acquisition/{k}",
                    op,
                    f"unsupported:{error}",
                    inputs=[ref.waveform_id],
                )
            )
            continue
        if not variant.ok:
            failed = ",".join(
                c["name"] for c in variant.checks if not c["pass"]
            )
            ledger.append(
                rejection(
                    unit,
                    f"acquisition/{k}",
                    op,
                    f"invalid_signal:{failed}",
                    inputs=[variant.ref.waveform_id],
                    recipe=recipe,
                )
            )
            continue
        writer.put_signal(variant.ref, variant.x)
        waveforms.append(variant.ref)
        vev = extractor.extract(
            variant.x, variant.ref
        ) + acquisition.inherit_annotations(ev, variant, extractor)
        evidence.extend(vev)
        for task in carried_from:
            carried, decision, cited = acquisition.carry(
                task, variant, vev, context
            )
            evidence.extend(cited)
            if carried is None:
                if decision != "not_carried":
                    ledger.append(
                        rejection(
                            unit,
                            f"acquisition/{k}",
                            task.family,
                            decision,
                            inputs=[variant.ref.waveform_id],
                            parent_task=task.task_id,
                            op=op,
                        )
                    )
                continue
            if decision.startswith("rejected"):
                ledger.append(ledger_row(carried, unit, f"acquisition/{k}"))
                continue
            record(carried, f"acquisition/{k}")
    if not cfg.pairs.variations or rng.random() >= cfg.pairs.variation_fraction:
        return
    spec = pairs.variation(
        cfg.pairs.variations[int(rng.integers(len(cfg.pairs.variations)))]
    )
    recipe = {
        "generator": f"{acquisition.GENERATOR}@{acquisition.VERSION}",
        **spec.recipe,
        "seed_key": int(rng.integers(2**53)),
        "variation": spec.name,
    }
    variant = acquisition.apply(x, ref, recipe, acq)
    if not variant.ok:
        ledger.append(
            rejection(
                unit,
                "variation",
                spec.name,
                "invalid_signal",
                inputs=[variant.ref.waveform_id],
            )
        )
        return
    vev = extractor.extract(variant.x, variant.ref)
    pair = pairs.check_pair(spec, View(ref, ev), View(variant.ref, vev))
    if pair.status != "accepted":
        ledger.append(
            rejection(
                unit,
                "variation",
                spec.name,
                pair.reason,
                inputs=[ref.waveform_id, variant.ref.waveform_id],
                pair=pair.to_json(),
            )
        )
        return
    writer.put_signal(variant.ref, variant.x)
    waveforms.append(variant.ref)
    evidence.extend(vev)
    accepted, rejected, cited = pairs.comparison_tasks(
        View(ref, ev),
        View(variant.ref, vev),
        [context, context],
        rng,
        pair=pair,
        attributes=list(spec.changes),
    )
    evidence.extend(cited)
    for task in accepted:
        record(task, "variation_pair")
    for task in rejected:
        ledger.append(ledger_row(task, unit, "variation_pair"))


def _generated_unit(
    name: str,
    bundles: list[Path],
    cfg: BuildConfig,
    source: QASource,
    extractor: EvidenceExtractor,
    quotas: Quotas,
    writer: CorpusWriter,
) -> tuple[list[QATask], list[Evidence], list[dict[str, Any]], list[InputRef]]:
    gen = cfg.generated
    tasks: list[QATask] = []
    evidence: list[Evidence] = []
    ledger: list[dict[str, Any]] = []
    waveforms: list[InputRef] = []
    samples = [
        sample for bundle in bundles for sample in generated.load_bundle(bundle)
    ]
    target = SignalInfo(
        source.signal.channels,
        source.signal.sample_rate,
        source.signal.units,
        source.signal.num_samples,
    )
    subject_of = source.subject_of
    # Ancestry first: no waveform is read for a sample whose conditioning
    # sources are outside the role.
    eligible = []
    for sample in samples:
        roots = sample.source_record_ids
        try:
            source.check_ancestry(
                [
                    (
                        sample.record_id,
                        roots,
                        tuple(subject_of.get(r, "") for r in roots),
                    )
                ]
            )
        except Exception as error:
            ledger.append(
                rejection(
                    name,
                    "ancestry",
                    "generated",
                    f"ancestry:{type(error).__name__}",
                    inputs=[sample.record_id],
                )
            )
            continue
        eligible.append(sample)
    verified: dict[str, tuple[View, generated.GeneratedSample]] = {}
    allowed = set(gen.required) | {
        "duration_s",
        "flat_leads",
        "available_leads",
        "channels",
        "sample_rate_hz",
    }
    # Regularity is derived from the same measured RR coefficient; its template
    # cites
    # that coefficient as rationale support, not as an independently accepted
    # target.
    if "rr_regularity" in gen.required:
        allowed.add("rr_cv")
    evol_cfg = EvolConfig(
        **{
            **asdict(cfg.evol),
            "seed_templates": list(gen.templates),
            "operators": ["breadth", "situation"],
        }
    )
    for sample in eligible:
        try:
            ref, x = generated.to_input(
                sample, target, subject_of, allow_resample=gen.allow_resample
            )
        except generated.GeneratedInputError as error:
            ledger.append(
                rejection(
                    name,
                    "convert",
                    "generated",
                    f"convert:{error}",
                    inputs=[sample.record_id],
                )
            )
            continue
        ref, ev, review = generated.verify_generated(
            ref, x, extractor, gen.required, min_confidence=gen.min_confidence
        )
        if review["status"] != "verified":
            ledger.append(
                rejection(
                    name,
                    "verify",
                    "generated",
                    "unverified:" + ",".join(review["reasons"]),
                    inputs=[ref.waveform_id],
                )
            )
            continue
        writer.put_signal(ref, x)
        waveforms.append(ref)
        evidence.extend(ev)
        view = View(ref, ev)
        verified[ref.waveform_id] = (view, sample)
        grown = evolve_waveform(
            view,
            ev,
            {"clinical_context": [student_context("generated", None)]},
            evol_cfg,
            cfg.seed,
        )
        evidence.extend(grown.evidence)
        store = {e.evidence_id: e for e in grown.evidence}
        for task in grown.accepted:
            attributes = {
                store[eid].attribute
                for eid in task.required_evidence
                if eid in store
            }
            if not attributes <= allowed:
                task = _reject(task, "attribute_not_verified_for_generated")
            elif not quotas.allow(task):
                task = _reject(task, "plan_quota_reached")
            ledger.append(ledger_row(task, name, "generated"))
            if task.verification["status"] == "accepted":
                tasks.append(task)
    if gen.heart_rate_pairs:
        _generated_pairs(name, verified, cfg, tasks, evidence, ledger)
    return tasks, evidence, ledger, waveforms


def _generated_pairs(name, verified, cfg, tasks, evidence, ledger) -> None:
    """Compare remeasured samples that differ only in requested heart rate."""
    by_seed: dict[int, list[tuple[View, Any]]] = defaultdict(list)
    for view, sample in verified.values():
        by_seed[sample.seed].append((view, sample))
    spec = pairs.variation("diffusets.heart_rate_condition@1")
    for seed_value, group in sorted(by_seed.items()):
        group.sort(key=lambda item: item[1].conditioning.get("heart_rate", 0))
        for (va, sa), (vb, sb) in itertools.pairwise(group):
            rest_a = {
                k: v for k, v in sa.conditioning.items() if k != "heart_rate"
            }
            rest_b = {
                k: v for k, v in sb.conditioning.items() if k != "heart_rate"
            }
            if rest_a != rest_b or sa.conditioning.get(
                "heart_rate"
            ) == sb.conditioning.get("heart_rate"):
                continue
            pair = pairs.check_pair(
                spec, va, vb, {"heart_rate_bpm": "increase"}
            )
            if pair.status != "accepted":
                ledger.append(
                    rejection(
                        name,
                        "generated_pair",
                        spec.name,
                        pair.reason,
                        inputs=[va.ref.waveform_id, vb.ref.waveform_id],
                        pair=pair.to_json(),
                    )
                )
                continue
            rng = stream(cfg.seed, "pair", va.ref.waveform_sha256, seed_value)
            accepted, rejected, cited = pairs.comparison_tasks(
                va, vb, ["", ""], rng, pair=pair, attributes=["heart_rate_bpm"]
            )
            evidence.extend(cited)
            for task in [*accepted, *rejected]:
                ledger.append(ledger_row(task, name, "generated_pair"))
            tasks.extend(accepted)


def _task_ids(cfg: BuildConfig) -> set[str] | None:
    """Load selected task identifiers after verifying their checksum."""
    if not cfg.task_ids_file:
        return None
    path = Path(cfg.task_ids_file)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not cfg.task_ids_sha256 or digest != cfg.task_ids_sha256:
        raise PermissionError(
            f"{path} has sha256 {digest[:12]}, not the pinned "
            f"{str(cfg.task_ids_sha256)[:12]}"
        )
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return set(
            pq.read_table(path, columns=["task_id"])
            .column("task_id")
            .to_pylist()
        )
    return set(json.loads(path.read_text(encoding="utf-8")))


def _reject(task: QATask, reason: str) -> QATask:
    from dataclasses import replace

    return replace(
        task,
        verification={
            **task.verification,
            "status": "rejected",
            "reason": reason,
        },
    )
